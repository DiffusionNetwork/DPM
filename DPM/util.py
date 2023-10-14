import shutil

import numpy as np
from param import *
import os

precision = 100000000           # 浮点数保留的位数

def load_record_states(path):
    """
    加载爆发记录
    :param path:
    :return:
    """
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        record_states = np.array([[int(node) for node in line] for line in lines])
        return record_states


def load_record_times(path):
    """
    加载爆发时间片
    :param path:
    :return:
    """
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        record_times = np.array([[int(time) for time in line] for line in lines])
        return record_times


def load_edges(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        lines = [[(int(node) - 1) for node in line] for line in lines]
        return lines


def load_community(node_num, com_num):
    path = pwd_path + 'dataset/{0}-{1}-4-0.3/community.dat'.format(node_num, com_num)
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [int(line.split()[1]) for line in lines]
        A = np.zeros((node_num, com_num), dtype=np.int)
        for i in range(node_num):
            A[i, lines[i] - 1] = 1
    return A


def float_round(number: float):
    return round(number * precision) / precision


def get_network_structure(node_num, edges):
    network_structure = np.zeros((node_num, node_num), dtype=np.int)
    for edge in edges:
        network_structure[edge[0], edge[1]] = 1
    return network_structure


# 清空指定类型方法的结果文件
def clear_synthetic_result_dir(method_name: str):
    node_num_list = [100, 150, 200, 250, 300]
    average_node_degree_list = [2, 3, 4, 5, 6]
    degree_dispersion_list = [1, 1.5, 2, 2.5, 3]
    community_size_distribution_list = [2]

    initial_infect_ratio_list = [0.05, 0.1, 0.15, 0.2, 0.25]
    diffusion_processes_number_list = [100, 150, 200, 250, 300]
    propagation_probability_list = [0.2, 0.25, 0.3, 0.35, 0.4]

    parameters = [node_num_list, average_node_degree_list,
                  degree_dispersion_list, community_size_distribution_list,
                  initial_infect_ratio_list, diffusion_processes_number_list,
                  propagation_probability_list]

    node_num = 200
    average_node_degree = 4
    degree_dispersion = 2
    community_size_distribution = 2

    initial_infect_ratio = 0.15
    diffusion_processes_number = 200
    propagation_probability = 0.3

    index_range = range(len(parameters))

    for i in index_range:
        parameter = [node_num, average_node_degree, degree_dispersion,
                     community_size_distribution, initial_infect_ratio,
                     diffusion_processes_number, propagation_probability]
        for j in range(len(parameters[i])):  # 对第i个参数进行遍历
            parameter[i] = parameters[i][j]
            dir_path = f'dataset/{parameter[0]}-{parameter[1]}-{parameter[2]}-{parameter[3]}-{parameter[4]}-{parameter[5]}-{parameter[6]}/' \
                       f'result/{method_name}'
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)     # 删除文件夹
            os.mkdir(dir_path)          # 建立文件夹


def clear_real_result_dir(method_name: str):
    dataset_list = ["NetSci", "DUNF"]
    initial_infect_ratio_list = [0.05, 0.1, 0.15, 0.2, 0.25]
    diffusion_processes_number_list = [100, 150, 200, 250, 300]
    propagation_probability_list = [0.2, 0.25, 0.3, 0.35, 0.4]

    parameters = [initial_infect_ratio_list, diffusion_processes_number_list,
                  propagation_probability_list]

    initial_infect_ratio = 0.15
    diffusion_processes_number = 200
    propagation_probability = 0.3

    index_range = range(len(parameters))

    for dataset in dataset_list:
        for i in index_range:
            parameter = [initial_infect_ratio, diffusion_processes_number, propagation_probability]
            for j in range(len(parameters[i])):  # 对第i个参数进行遍历
                parameter[i] = parameters[i][j]
                dir_path = pwd_path + f'dataset/{dataset}/{parameter[0]}-{parameter[1]}-{parameter[2]}/' \
                                      f'result/{method_name}'
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)  # 删除文件夹
                os.mkdir(dir_path)  # 建立文件夹


def get_decimal_place(num):
    place = 0
    while num < 1:
        num *= 10
        place += 1
    return place



import random
import numpy as np

def generateData(network: np.ndarray, p_matrix: np.ndarray, infectRate=0, seed=None):
    node_num = network.shape[0]
    node_state = np.zeros(node_num, dtype=np.int)

    # 选取随机种子
    if infectRate != 0:
        seed_infected_node_num = infectRate * node_num
        while seed_infected_node_num > 0:
            selected_node = random.randint(0, node_num)
            if node_state[selected_node] == 0:
                node_state[selected_node] = 1
                seed_infected_node_num -= 1
    else:
        for s in seed:
            node_state[s] = 1

    # 模拟爆发
    break_flag = False
    new_infected_nodes = np.where(node_state == 1)[0].tolist()
    while not break_flag:
        # 感染邻居节点
        new_infected_neighbors = []
        for node in new_infected_nodes:
            neighbor_nodes = np.where(network[node] == 1)[0]
            for neighbor_node in neighbor_nodes:
                if node_state[neighbor_node] == 0:
                    if random.random() < p_matrix[node][neighbor_node]:
                        node_state[neighbor_node] = 1
                        new_infected_neighbors.append(neighbor_node)
        if np.sum(new_infected_nodes) == 0:
            break_flag = True
        new_infected_nodes = new_infected_neighbors

    return node_state

def stimulate_prob(network: np.ndarray, p_matrix: np.ndarray, break_num: int, save_path=None):
    if (save_path is not None) and os.path.exists(save_path):
        return np.loadtxt(save_path, delimiter=',', dtype=np.float)
    node_num = network.shape[0]
    # break_num = 1000
    pc_matrix = np.zeros((node_num, node_num))
    for i in range(node_num):
        # print(f"\t{i}/{node_num}")
        for j in range(break_num):
            pc_matrix[i] += generateData(network, p_matrix, seed=[i])
    pc_matrix /= break_num
    if save_path is not None:
        np.savetxt(save_path, pc_matrix, fmt='%f', delimiter=',')
    return pc_matrix
