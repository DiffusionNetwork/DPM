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
