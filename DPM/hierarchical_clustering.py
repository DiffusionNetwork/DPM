import time
import numpy as np
from dset import get_eu_dist_matrix
from dom_sets import get_homop_sim_matrix, get_cosine_sim_matrix, get_man_sim_matrix, get_eu_sim_matrix

def get_com_dist_matrix(node_dist_matrix: np.ndarray, A: np.ndarray, mode=3):
    node_num, com_num = A.shape
    com_dist_matrix = np.zeros((com_num, com_num))
    for com1 in range(com_num):
        for com2 in range(com_num):
            if com1 > com2:
                com_dist_matrix[com1][com2] = com_dist_matrix[com2][com1]
                continue
            elif com1 == com2:
                com_dist_matrix[com1][com2] = np.inf    # 对角线距离置为无穷大，避免找距离最近团找到自己
                continue
            com1_nodes = np.where(A[:, com1] != 0)[0]
            com2_nodes = np.where(A[:, com2] != 0)[0]
            tmp_dist_matrix = node_dist_matrix[com1_nodes][:, com2_nodes]
            if mode == "Single":       # Single Linkage
                com_dist_matrix[com1][com2] = np.min(tmp_dist_matrix)
            elif mode == "Complete":
                com_dist_matrix[com1][com2] = np.max(tmp_dist_matrix)
            else:
                com_dist_matrix[com1][com2] = np.average(tmp_dist_matrix)
    return com_dist_matrix

def hier_cluster(eivec_matrix: np.ndarray, target_com_num: int, com_ratio_list: list, mode=3, dist_mode="Homop"):
    node_num = eivec_matrix.shape[0]
    # node_dist_matrix = get_eu_dist_matrix(eivec_matrix)
    node_dist_matrix = 1 - get_homop_sim_matrix(eivec_matrix)
    if dist_mode == "Cos":
        node_dist_matrix = get_cosine_sim_matrix(eivec_matrix)
    elif dist_mode == "Man":
        node_dist_matrix = get_man_sim_matrix(eivec_matrix)
    elif dist_mode == "EU":
        node_dist_matrix = get_eu_dist_matrix(eivec_matrix)

    A = np.eye(node_num, dtype=np.int)  # 初始团矩阵
    A_results = []
    time_results = []

    com_num = node_num  # 开始时每个节点都是一个团
    ratio_list = com_ratio_list.copy()
    start = time.time()

    while len(ratio_list) != 0:
        com_dist_matrix = get_com_dist_matrix(node_dist_matrix, A, mode)
        min_index = np.where(com_dist_matrix == np.min(com_dist_matrix))  # 选择距离最小的团聚合
        com_x = min_index[0][0]
        com_y = min_index[1][0]
        A[:, com_x] += A[:, com_y]
        A = np.delete(A, com_y, axis=1)  # 将团y合并入团x, 然后删除团y
        com_num -= 1
        print('com_num = ' + str(com_num) + ' / ' + str(target_com_num))
        if com_num == target_com_num:
            end = time.time()
            A_results.insert(0, A.copy())
            time_results.insert(0, end - start)
            break


        # for ratio in ratio_list:
        #     assert 1 < int(target_com_num * ratio) < node_num
        #     if int(target_com_num * ratio) == com_num:
        #         end = time.time()
        #         A_results.insert(0, A.copy())
        #         time_results.insert(0, end - start)
        #         ratio_list.remove(ratio)
        #         break

    return A_results, time_results
