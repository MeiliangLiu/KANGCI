import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import ParameterGrid
from ComputeROC import compute_roc
from src.efficient_kan import KAN
from tool import simulate_var, var_read

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#The VAR dataset is run only on the origin time series!!!

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


def infer_Grangercausality(P, T, lag, sparsity, epoch, hidden_size, lam, lam_ridge, learning_rate):
    # Set seed for random number generation (for reproducibility of results)
    global_seed = 1
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed(global_seed)
    np.random.seed(global_seed)
    score = 0

    best_score = 0
    total_score = 0

    X, GC = var_read(sparsity, lag, 1)  #modify the trial in here
    X = torch.tensor(X, dtype=torch.float32, device=device)

    train_x = X[:T - lag, :]
    train_y = X[lag:, :]

    #component-wise generate p models
    networks = []
    for _ in range(P):
        network = KAN([P, hidden_size, 1], base_activation=nn.Identity).to(device)
        networks.append(network)

    models = nn.ModuleList(networks)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(models.parameters(), lr=learning_rate)

    for i in range(epoch):
        losses1 = []
        for j in range(0, P):
            network_output = models[j](train_x).view(-1)
            loss_i = loss_fn(network_output, train_y[:, j])
            losses1.append(loss_i)

        predict_loss1 = sum(losses1)

        ridge_loss1 = sum([ridge_regularize(model, lam_ridge) for model in models[:P]])
        regularize_loss1 = sum([regularize(model, lam, "GL", learning_rate) for model in models[:P]])

        loss = predict_loss1 + regularize_loss1 + ridge_loss1

        GCs = []
        for k in range(P):
            GCs.append(models[k].GC().detach().cpu().numpy())
        GCs = np.array(GCs)

        score1 = compute_roc(GC, GCs, False)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if best_score < score1 and score1 > 0.99:
            best_score = score1

        total_score += score
        if (i + 1) % 1 == 0:
            print(
                f'Epoch [{i + 1}/{epoch}], predict_loss: {predict_loss1.item():.4f}, '
                f'regularize_loss: {regularize_loss1.item():.4f}, '
                f'ridge_loss :{ridge_loss1.item():.4f}, '
                f'score: {score1:.4f}')
    print('Score:' + str(best_score))
    return score


def grid_search(param_grid):
    results = []
    param_list = list(ParameterGrid(param_grid))

    for params in param_list:
        print(f"Training with params: {params}")

        avg_score = infer_Grangercausality(10, 1000, 3, 0.2, 50,
                                           hidden_size=params['hidden_size'], lam=params['lam'],
                                           lam_ridge=params['lam_ridge'], learning_rate=params['learning_rate']
                                           )
        results.append((params, avg_score))

    best_params = max(results, key=lambda x: x[1])
    print(f"Best params: {best_params[0]} with avg score: {best_params[1]}")
    return best_params


if __name__ == '__main__':
    # param_grid = {
    #     'hidden_size': [8,10,12,16,20,32,36,40,50,100,200],
    #     'lam': [0.1,0.01,0.001,0.0001],
    #     'lam_ridge': [10,12,16,20,24,25,30,32],
    #     'learning_rate': [0.01,0.001]
    # }

    param_grid = {
        'hidden_size': [16],
        'lam': [0.0001],
        'lam_ridge': [0.05],
        'learning_rate': [0.01]
    }

    best_params = grid_search(param_grid)
