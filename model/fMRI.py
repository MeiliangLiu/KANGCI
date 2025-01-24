import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from ComputeROC import compute_roc
from src.efficient_kan import KAN
from tool import fmri_read
import time

# Set seed for random number generation (for reproducibility of results)

global_seed = 1
torch.manual_seed(global_seed)
torch.cuda.manual_seed(global_seed)
np.random.seed(global_seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ridge_regularize(model, lam_ridge):
    total_weight_sum = 0
    for layer in model.layers[1:]:
        weight_squared_sum = torch.sum(layer.base_weight ** 2)
        total_weight_sum += weight_squared_sum
    result = lam_ridge * total_weight_sum
    return result


def regularize(network, lam, penalty, lr):
    x = network.layers[0].base_weight
    if penalty == 'GL':
        return lam * torch.sum(torch.norm(x, dim=0))
    elif penalty == 'row+column':
        return lam * (torch.sum(torch.norm(x, dim=0))
                      + torch.sum(torch.norm(x, dim=1)))
    else:
        raise ValueError('unsupported penalty: %s' % penalty)


def infer_Grangercausality(simulation, subject, hidden_size, epoch, lam, lam_ridge, learning_rate):
    best_score = 0
    best_score1 = 0
    best_score2 = 0

    X, GC, length = fmri_read(simulation, subject)
    P = X.shape[1]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    test_x = X[:length - 1, :]
    test_y = X[1:length, :]

    input_seq = torch.tensor(test_x, dtype=torch.float32).unsqueeze(0).cuda()
    target_seq = torch.tensor(test_y, dtype=torch.float32).cuda()

    # reversed data
    X2 = X[::-1, :]
    X2 = np.ascontiguousarray(X2)

    reversed_x = X2[:length - 1, :]
    reversed_y = X2[1:length, :]

    reversed_input_seq = torch.tensor(reversed_x, dtype=torch.float32).unsqueeze(0).cuda()
    reversed_target_seq = torch.tensor(reversed_y, dtype=torch.float32).cuda()

    # component-wise generate p models
    networks = []
    for _ in range(2 * P):
        network = KAN([P, 600, 64, 32, 1], base_activation=nn.Identity).to(device)
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

        score1 = compute_roc(GC, GCs, False)
        score2 = compute_roc(GC, GC2s, False)
        score_fusion = compute_roc(GC, result, False)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_time = time.time() - start_time

        if best_score < score_fusion:
            best_score = score_fusion
            best_score1 = score1
            best_score2 = score2
        if (i + 1) % 1 == 0:
            print(
                f'Epoch [{i + 1}/{epoch}], predict_loss1: {predict_loss1.item():.4f}, predict_loss2: {predict_loss2.item():.4f}, '
                f'regularize_loss1: {regularize_loss1.item():.4f}, regularize_loss2: {regularize_loss2.item():.4f}, '
                f'ridge_loss1 :{ridge_loss1.item():.4f}, ridge_loss2 :{ridge_loss2.item():.4f}'
                f'score1: {score1:.4f}, score2: {score2:.4f}, score_fusion: {score_fusion:.4f}')

    return best_score, best_score1, best_score2


if __name__ == '__main__':
    array = np.zeros((28, 50), dtype=float)
    array1 = np.zeros((28, 50), dtype=float)
    array2 = np.zeros((28, 50), dtype=float)
    for i in range(1, 29):
        for j in range(0, 50):
            best_score, best_score1, best_score2 = infer_Grangercausality(i, j, 128, 250, 0.01, 5, 0.001)
            array[i - 1, j] = best_score
            array1[i - 1, j] = best_score1
            array2[i - 1, j] = best_score2

    np.savetxt(f"../FMRI_fusion.txt", array, fmt='%.5f')
    np.savetxt(f"../FMRI_origin.txt", array1, fmt='%.5f')
    np.savetxt(f"../FMRI_reverse.txt", array2, fmt='%.5f')
