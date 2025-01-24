import csv
import numpy as np
import scipy.io as sio


def dream_read_label(file_path, p):
    label = np.zeros((p, p))
    with open(file_path, "r") as file:
        # 逐行读取文件内容
        for line in file:
            lines = line.strip().split("\t")  # 去除行尾的换行符和空格
            # print(lines)
            label[int(lines[1].replace("G", "")) - 1][int(lines[0].replace("G", "")) - 1] = lines[2]
    return label


def dream_read_data(file_path):
    data = []
    batch = []
    # 打开文件
    with open(file_path, "r") as file:
        # 创建 CSV 读取器，设置分隔符为制表符
        reader = csv.reader(file, delimiter="\t")
        # 逐行读取文件内容
        for i, row in enumerate(reader):
            if i == 0:
                continue
            # 文件中的21行通常是空行
            if i % 22 == 0:
                data.append(batch)
                batch = []
            else:
                float_list = [float(x) for x in row[1:]]
                batch.append(float_list)
        data.append(batch)
    return np.array(data)


def var_read(sparsity, lag, trial):
    data = sio.loadmat(
        r'C:\Users\admin\Desktop\efficient-kan\VAR\sparsity=' + str(sparsity) + ', lag=' + str(lag) + '\data\sim' + str(
            trial) + '.mat')
    GC = sio.loadmat(r'C:\Users\admin\Desktop\efficient-kan\VAR\sparsity=' + str(sparsity) + ', lag=' + str(lag) +
                     '\TrueGC\GC_' + str(trial) + '.mat')
    data = data['data']
    GC = GC['GC']
    return data, GC


def fmri_read(simulation, subject):
    data = sio.loadmat(r'C:\Users\admin\Desktop\efficient-kan\fmri bold\sim' + str(simulation) + '.mat')
    net = data['net'][subject]
    GC = np.where(net != 0, 1, 0)
    ts = data['ts']

    channel = 0
    length = 0
    nodes_5 = {1, 5, 7, 8, 9, 10, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23,
               24, 25, 26, 27, 28}
    nodes_10 = {2, 6, 11, 12, 17}
    nodes_15 = {3}
    nodes_50 = {4}

    length_50 = {26, 27}
    length_100 = {25, 28}
    length_200 = {1, 2, 3, 4, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24}
    length_1200 = {5, 6}
    length_2400 = {19, 20}
    length_5000 = {7, 9}

    if simulation in nodes_5:
        channel = 5
    elif simulation in nodes_10:
        channel = 10
    elif simulation in nodes_15:
        channel = 15
    elif simulation in nodes_50:
        channel = 50

    if simulation in length_50:
        length = 50
    elif simulation in length_100:
        length = 100
    elif simulation in length_200:
        length = 200
    elif simulation in length_1200:
        length = 1200
    elif simulation in length_2400:
        length = 2400
    elif simulation in length_5000:
        length = 5000

    X = ts.reshape((50, length, channel))[subject]
    # data label
    return X, GC, length


def make_var_stationary(beta, radius=0.97):
    '''Rescale coefficients of VAR a_model to make stable.'''
    p = beta.shape[0]
    lag = beta.shape[1] // p
    bottom = np.hstack((np.eye(p * (lag - 1)), np.zeros((p * (lag - 1), p))))
    beta_tilde = np.vstack((beta, bottom))
    eigvals = np.linalg.eigvals(beta_tilde)
    max_eig = max(np.abs(eigvals))
    nonstationary = max_eig > radius
    if nonstationary:
        return make_var_stationary(0.95 * beta, radius)
    else:
        return beta


def simulate_var(p, T, lag, sparsity, beta_value=1.0, sd=0.1):
    np.random.seed()
    # Set up coefficients and Granger causality ground truth.建立系数和格兰杰因果关系的基本事实
    GC = np.eye(p, dtype=int)  # 生成一个P*P的对角线为1，其余都是0的矩阵
    # beta是一个与GC一样的矩阵但值是beta_value
    beta = np.eye(p) * beta_value  # 生成一个P*P的对角线为1beta_value
    # sparsity稀疏性
    num_nonzero = int(p * sparsity) - 1
    # 决定因果关系
    for i in range(p):
        choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        beta[i, choice] = beta_value
        GC[i, choice] = 1
    # beta在水平方向上叠加了lag个 [10,10]叠加3次=>[10,30]
    beta = np.hstack([beta for _ in range(lag)])
    # 重新缩放VAR模型的系数以使其稳定，beta内除0以外的值就变了
    beta = make_var_stationary(beta)

    # Generate data.开始生成数据
    burn_in = 100  # 开始的100个生成点数据不要了
    errors = np.random.normal(scale=sd, size=(p, T + burn_in))  # [10,1000]
    X = np.zeros((p, T + burn_in))
    X[:, :lag] = errors[:, :lag]
    for t in range(lag, T + burn_in):
        X[:, t] = np.dot(beta, X[:, (t - lag):t].flatten(order='F'))  # 计算前lag与beta的点积
        X[:, t] += + errors[:, t - 1]

    return X.T[burn_in:], GC
