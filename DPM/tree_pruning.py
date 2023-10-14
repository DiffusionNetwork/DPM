from cal_score import *
from sklearn.cluster import KMeans
import random


def cal_pruning_matrix(data, sigma):
    # print("begin calculating HSIC matrix")
    cov = cal_hsic_matrix(data, sigma=sigma)
    # print("done")
    cov2 = copy.deepcopy(cov)
    min_v = np.min(cov2)
    for i in range(cov2.shape[0]):
        cov2[i, i] = min_v
    cov2 = np.reshape(cov2, (-1, 1))
    km = KMeans(2)
    km.fit(cov2)
    pruning_matrix = np.reshape(km.labels_, (cov.shape[0], cov.shape[1]))
    # pruning_matrix = np.reshape(k_means(np.reshape(cov2, (-1, 1))), (cov.shape[0], cov.shape[1]))
    return pruning_matrix, cov


def cal_pow_set(set):
    cardinality = len(set)
    powset = []
    for i in range(2**cardinality):
        binary_i = bin(i)[2:].zfill(cardinality)
        subset = []
        for index, bit in enumerate(reversed(binary_i)):
            if bit == '1':
                subset.append(set[index])
        powset.append(subset)
    return powset


def k_means(data, n_clusters=2, max_iter=50):
    """
    只针对一维数据的聚类
    :param data:  n*1 narray
    :param n_clusters:
    :param max_iter:
    :return:
    """
    centroids = [0, 0]
    centroids_label = []
    while centroids[0] == centroids[1]:
        centroids = []
        centroids_label = []
        # 初始化中心,簇标签
        for i in range(n_clusters):
            centroids.append(data[random.randint(0, data.shape[0]), :])
        centroids.sort()
        for i in range(n_clusters):
            centroids_label.append(i)
        centroids[0] = np.min(data)

    # for i in range(n_clusters):
    #     if i == 0:
    #         continue
    #     data_sum = 0
    #     count = 0
    #     for index, element in enumerate(data):
    #         data_sum += element
    #         count += 1
    #     centroids[i] = data_sum / count

    for iter_time in range(max_iter):
        # 计算每个点所属的簇
        _labels = []
        for element in data:
            best_label = -1
            shortest_distance = float('inf')
            for label, centroid in enumerate(centroids):
                distance = np.abs(element - centroid)
                if distance < shortest_distance:
                    shortest_distance = distance
                    best_label = label
            _labels.append(best_label)

        # 更新质心,不更新最小的质心
        for i in range(n_clusters):
            if i == 0:
                continue
            data_sum = 0
            count = 0
            for index, element in enumerate(data):
                if _labels[index] == i:
                    data_sum += element
                    count += 1
            centroids[i] = data_sum / count
    return np.array(_labels)

