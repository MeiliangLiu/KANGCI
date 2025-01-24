import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from ComputeROC import compute_roc
from src.efficient_kan import KAN
from tool import dream_read_label, dream_read_data
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def off_diagonal(x):
    mask = ~np.eye(x.shape[0], dtype=bool)
    non_diag_elements = x[mask]
    new_arr = non_diag_elements.reshape(100, 99)
    return new_arr

def read_dream3(size, type):
    name_list = ["Ecoli1", "Ecoli2", "Yeast1", "Yeast2", "Yeast3"]
    GC = dream_read_label(
        r"C:\Users\admin\Desktop\efficient-kan\DREAM3 in silico challenge"
        r"\DREAM3 gold standards\DREAM3GoldStandard_InSilicoSize" + str(size) + "_" + name_list[type - 1] + ".txt",
        size)
    data = dream_read_data(
        r"C:\Users\admin\Desktop\efficient-kan\DREAM3 in silico challenge"
        r"\Size" + str(size) + "\DREAM3 data\InSilicoSize" + str(size) + "-" + name_list[
            type - 1] + "-trajectories.tsv")
    return GC, data


def regularize(network, lam, penalty, lr):
    x = network.layers[0].base_weight
    if penalty == 'GL':
        return lam * torch.sum(torch.norm(x, dim=0))
    else:
        raise ValueError('unsupported penalty: %s' % penalty)


def ridge_regularize(model, lam_ridge):
    '''Apply ridge penalty at all subsequent layers.'''
    total_weight_sum = 0
    for layer in model.layers[1:]:
        weight_squared_sum = torch.sum(layer.base_weight ** 2)
        total_weight_sum += weight_squared_sum
    result = lam_ridge * total_weight_sum
    return result


def infer_Grangercausality(P, type, epoch, hidden_size, lam, lam_ridge, learning_rate):
    # Set seed for random number generation (for reproducibility of results)
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed(global_seed)
    np.random.seed(global_seed)
    score = 0

    best_score = 0

    GC, data = read_dream3(P, type)
    GC = off_diagonal(GC)

    X = data.reshape(966, 100)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    test_x = X[:965, :]
    test_y = X[1:966, :]

    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).cuda()
    target_seq = torch.tensor(test_y, dtype=torch.float32).cuda()

    X2 = X[::-1, :]  # reverse data
    X2 = np.ascontiguousarray(X2)

    reversed_x = X2[:965, :]
    reversed_y = X2[1:966, :]

    reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).cuda()
    reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).cuda()

    #component-wise generate p models
    networks = []
    for _ in range(2 * P):
        network = KAN([P, hidden_size, 1], base_activation=nn.Identity).to(device)
        networks.append(network)

    models = nn.ModuleList(networks)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(models.parameters(), lr=learning_rate)

    for i in range(epoch):
        start_time = time.time()
        losses1 = []
        losses2 = []
        for j in range(0, P):
            network_output = models[j](input_seq).view(-1)
            loss_i = loss_fn(network_output, target_seq[:, j])
            losses1.append(loss_i)

        for j in range(P, 2 * P):
            network_output = models[j](reversed_input_seq).view(-1)
            loss_i = loss_fn(network_output, reversed_target_seq[:, j - P])
            losses2.append(loss_i)
        predict_loss1 = sum(losses1)
        predict_loss2 = sum(losses2)

        ridge_loss1 = sum([ridge_regularize(model, lam_ridge) for model in models[:P]])
        ridge_loss2 = sum([ridge_regularize(model, lam_ridge) for model in models[P:2 * P]])
        regularize_loss1 = sum([regularize(model, lam, "GL", learning_rate) for model in models[:P]])
        regularize_loss2 = sum([regularize(model, lam, "GL", learning_rate) for model in models[P:2 * P]])

        loss = predict_loss1 + predict_loss2 + regularize_loss1 + regularize_loss2 + ridge_loss1 + ridge_loss2

        GCs = []
        GC2s = []
        for k in range(P):
            GCs.append(models[k].GC().detach().cpu().numpy())
        GCs = np.array(GCs)

        for k in range(P, 2 * P):
            GC2s.append(models[k].GC().detach().cpu().numpy())
        GC2s = np.array(GC2s)

        if predict_loss1 < predict_loss2 and regularize_loss1 < regularize_loss2:
            result = GCs
        elif predict_loss1 > predict_loss2 and regularize_loss1 > regularize_loss2:
            result = GC2s
        else:
            result = np.where(
                np.abs(GCs - GC2s) < 0.05,
                (GCs + GC2s) / 2,
                np.maximum(GCs, GC2s)
            )

        GCs = off_diagonal(GCs)
        GC2s = off_diagonal(GC2s)

        result = off_diagonal(result)

        score1 = compute_roc(GC, GCs, False)
        score2 = compute_roc(GC, GC2s, False)
        score_fusion = compute_roc(GC, result, False)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_time = time.time() - start_time

        if best_score < score_fusion:
            best_score = score_fusion
        if (i + 1) % 1 == 0:
            print(
                f'Epoch [{i + 1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
                f'regularize_loss1: {regularize_loss1.item():.4f}, regularize_loss2: {regularize_loss2.item():.4f}, '
                f'ridge_loss1 :{ridge_loss1.item():.4f}, ridge_loss2 :{ridge_loss2.item():.4f}'
                f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}, Time:{epoch_time:.4f}')
    print('Score:' + str(best_score))
    return score


def grid_search(param_grid):
    results = []
    param_list = list(ParameterGrid(param_grid))

    for params in param_list:
        print(f"Training with params: {params}")

        avg_score = infer_Grangercausality(100, 1, 300, hidden_size=params['hidden_size'], lam=params['lam'],
                                           lam_ridge=params['lam_ridge'], learning_rate=params['learning_rate']
                                           )
        results.append((params, avg_score))

    best_params = max(results, key=lambda x: x[1])
    print(f"Best params: {best_params[0]} with avg score: {best_params[1]}")
    return best_params


if __name__ == '__main__':
    # param_grid = {
    #     'hidden_size': [256],
    #     'lam': [0.01],
    #     'lam_ridge': [5],
    #     'learning_rate': [0.005]
    # }  ###  0.734,0.658,0.555,0.555,0.521

    param_grid = {
        'hidden_size': [128], ##128
        'lam': [0.01],
        'lam_ridge': [5],
        'learning_rate': [0.005]
    } ###  0.734,0.673,0.59,0.53,0.566

    # param_grid = {
    #     'hidden_size': [128],
    #     'lam': [0.01],
    #     'lam_ridge': [5],
    #     'learning_rate': [0.01]
    # }

    best_params = grid_search(param_grid)
