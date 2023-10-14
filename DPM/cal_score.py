import numpy as np
import copy
from concurrent.futures import ThreadPoolExecutor, wait


def cal_f_score(true_matrix, est_matrix):
    true_edges = np.sum(true_matrix)
    est_edges = np.sum(est_matrix)
    correct_edges = 0
    for i in range(true_matrix.shape[0]):
        for j in range(true_matrix.shape[0]):
            if true_matrix[i, j] == 1 and est_matrix[i, j] == 1:
                correct_edges += 1
    precision = correct_edges/est_edges
    recall = correct_edges/true_edges
    return precision, recall


def cal_dist(x, y):
    x = np.reshape(x, (1, -1))
    y = np.reshape(y, (1, -1))
    num = np.dot(x, y.T)  # 若为行向量则 A * B.T
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return 0
    cos = num[0, 0] / denom
    return 1 - cos


def cal_k_matrix(x, sigma):
    """
    :param x: n*m的数组，n为记录条数，m为数据维度
    :param sigma:用于计算矩阵的常数
    :return:K矩阵
    """
    # data_size = x.shape[0]
    # k_matrix = np.zeros((data_size, data_size))
    # for i in range(data_size):
    #     for j in range(data_size):
    #         k_matrix[i, j] = np.exp(-np.linalg.norm(x[i]-x[j]) ** 2 / sigma ** 2)

    if len(x.shape) == 1:
        x = np.reshape(x, (x.shape[0], 1))
    x1 = np.dot(x, x.T)
    xd = np.expand_dims(x1.diagonal(), 0)
    i = np.ones((1, xd.shape[1]))
    x2 = np.dot(xd.T, i)
    x3 = np.dot(i.T, xd)
    Kx = x2 + x3 - 2*x1
    Kx = np.exp(- Kx / sigma ** 2)
    return Kx


def cal_hsic(kx, ky):
    n = kx.shape[0]
    # j = np.identity(n)
    # j -= 1/n
    # temp = np.dot(kx, j)
    # temp = np.dot(temp, ky)
    # temp = np.dot(temp, j)
    # return np.trace(temp)/(n-1)**2
    kxy = np.dot(kx, ky)
    h = np.trace(kxy) / n ** 2 + np.mean(kx) * np.mean(ky) - 2 * np.mean(kxy) / n
    return h * n ** 2 / (n - 1) ** 2


def update_hsic_matrix(hsic_matrix, i, j, kx, ky):
    hsic_matrix[i, j] = cal_hsic(kx, ky)


def cal_hsic_matrix(all_data, sigma):
    kx_list = []
    for i in range(all_data.shape[1]):
        # print(i)
        kx = cal_k_matrix(all_data[:, i], sigma)
        kx_list.append(kx)
    dimension = len(kx_list)
    hsic_matrix = np.zeros((dimension, dimension))
    # with ThreadPoolExecutor(max_workers=5) as t:
    #     all_task = [t.submit(spider, page) for page in range(1, 5)]
    #     wait(all_task, return_when=FIRST_COMPLETED)
    #     print('finished')
    #     print(wait(all_task, timeout=2.5))
    # for i in range(dimension):
    #     for j in range(dimension):
    #         hsic_matrix[i][j] = cal_hsic(kx_list[i], kx_list[j])
    with ThreadPoolExecutor(max_workers=5) as tp:
        all_task = []
        for i in range(dimension):
            for j in range(dimension):
                all_task.append(tp.submit(update_hsic_matrix, hsic_matrix, i, j, kx_list[i], kx_list[j]))
        wait(all_task)
    return hsic_matrix


def cal_score(x, y, sigma=1):
    kx = cal_k_matrix(x, sigma)
    ky = cal_k_matrix(y, sigma)
    return cal_hsic(kx, ky)/cal_hsic(ky, ky) - cal_bias(x, y, sigma, 1)


def cal_bias(x, y, sigma, n):
    """
    :param x: 父节点组合
    :param y: 目标节点
    :param sigma: 计算k矩阵的一个参数
    :param n: 对y进行重排列的次数
    :return: x，y之间的HSIC的误差
    """
    y_temp = copy.deepcopy(y)
    bias = 0
    for i in range(n):
        y_temp = np.random.permutation(y_temp)
        kx = cal_k_matrix(x, sigma)
        ky = cal_k_matrix(y_temp, sigma)
        bias += cal_hsic(kx, ky)/cal_hsic(ky, ky)
    return bias/n


def cal_upper_bound(x, y, sigma, n):
    return 1 - cal_bias(x, y, sigma, n) / np.exp(n/sigma)


def cal_f(p, r):
    return 2 / (1/p + 1/r)




