import random

import numpy as np
import torch
from scipy.integrate import odeint

def lorenz(x, t, F):
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = (x[(i+1) % p] - x[(i-2) % p]) * x[(i-1) % p] - x[i] + F

    return dxdt

def simulate_lorenz_96(p, T, F=10.0, delta_t=0.1, sd=0.1, burn_in=1000,
                       seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Use scipy to solve ODE.
    x0 = np.random.normal(scale=0.01, size=p)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    # 使用SciPy库的odeint函数求解了洛伦兹系统
    X = odeint(lorenz, x0, t, args=(F,))
    X += np.random.normal(scale=sd, size=(T + burn_in, p))

    # Set up Granger causality ground truth.
    GC = np.zeros((p, p), dtype=int)
    for i in range(p):
        GC[i, i] = 1
        GC[i, (i + 1) % p] = 1
        GC[i, (i - 1) % p] = 1
        GC[i, (i - 2) % p] = 1

    return X[burn_in:], GC


def data_segmentation(data,lag=5, seg=1, val_rate = 0.2):
    T = len(data)
    data_x = []
    train_input = data[:T - 1, :]
    for i in range(0, train_input.shape[0] - lag + 1):
        x = train_input[i:i + lag, :]
        x = x.transpose(1, 0)
        data_x.append(x)
    data_x = torch.stack(data_x)
    data_y = data[lag:T, :]

    train_x = []
    train_y = []
    val_x = []
    val_y = []
    for n in range(seg):
        a = n * int(T / seg)
        b = (n + 1) * int(T / seg)
        x = data_x[a:b - int(val_rate * T / seg)]
        train_x.append(x)
        y = data_y[a:b - int(val_rate * T / seg)]
        train_y.append(y)
        x = data_x[b - int(val_rate * T / seg):b]
        val_x.append(x)
        y = data_y[b - int(val_rate * T / seg):b]
        val_y.append(y)
    if seg == 1:
        return train_x[0], train_y[0], val_x[0], val_y[0]
    else:
        return train_x, train_y, val_x, val_y

