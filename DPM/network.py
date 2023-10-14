import time
from tree_pruning import *
from BAB import *


def construct_network_hsic(record_states, sigma=20, permutation_times=1):

    pruning_matrix, cov = cal_pruning_matrix(record_states, sigma=sigma)
    # return pruning_matrix, cov

    node_num = record_states.shape[1]
    est_matrix = np.zeros((node_num, node_num))
    for i in range(node_num):
        # print(f"\t\t{i}/{node_num}")
        y_data = record_states[:, i]
        data_size = record_states.shape[0]
        x_data = np.zeros((data_size, 1))
        node_index = [i]
        for parent in range(record_states.shape[1]):
            if pruning_matrix[parent, i] == 1:
                x_data = np.hstack((x_data, np.reshape(record_states[:, parent], (-1, 1))))
                node_index.append(parent)
        # print("node_index ", i, ": ", node_index)
        if x_data.shape[1] == 1:
            # print("no parents for ", i)
            # print("continue next loop")
            continue
        x_data = x_data[:, 1:]
        y_data = np.reshape(y_data, (-1, 1))
        all_data = np.hstack((y_data, x_data))
        parents, best_score, history_parents, node_count, pruning_count = branch_and_bound(all_data, target_node=0,
            score=-1, sigma=sigma, permutation_times=permutation_times)
        # print("parents:", parents)
        # print("score:", best_score)
        # print("history parents: ")
        # for hp in sorted(history_parents):
        #     print((hp, history_parents[hp]))
        for parent in parents:
            est_matrix[node_index[parent], i] = 1
        # print("assessment ", i, " done")

    return est_matrix, cov



def net_diffusion_rate_m(record_states, network_structure):
    # network diffusion rate estimation
    # 传播网络传播概率估计
    # 输入： record_states(np.array)-感染传播结果，network_structure(np.array)-网络结构
    # 输出： 每条边对应的感染概率

    record_num, node_num = record_states.shape
    f = np.zeros([record_num, node_num, node_num])    # f 对应论文中的phi
    p_matrix = np.random.random([node_num, node_num])  # 随机初始化传播概率
    max_err = 0.001   # 迭代终止条件
    iter_cnt = 0
    neg_cnt = np.zeros([node_num, node_num])  # 论文中的|S^-|

    # start pre-calculation   # 预先对爆发结果进行统计，得到之后EM迭代需要用到的中间变量
    node_positive_record_mask = np.zeros([node_num, node_num, record_num])   # node_positive_record_mask[i,j,L] = 1，表示在第L条爆发记录中，节点i,j都处于感染状态
    for i in range(node_num):
        for j in range(node_num):
            if network_structure[i, j] == 0:
                continue

            for index in range(record_num):
                if record_states[index, i] == 1 and record_states[index, j] == 1:
                    node_positive_record_mask[i, j, index] = 1
                elif record_states[index, i] != record_states[index, j]:
                    neg_cnt[i, j] += 1
    # end pre-calculation

    while True:
        iter_cnt += 1

        # first: update f
        tmp_1_p = 1 - p_matrix
        for idx in range(record_num):
            graph_cas = network_structure * record_states[idx].reshape([node_num, 1])
            tmp = tmp_1_p * graph_cas
            tmp[tmp == 0] = 1
            sub = 1 - np.prod(tmp, axis=0)
            sub[sub == 0] = np.inf
            f[idx, :, :] = np.copy(p_matrix / sub)
            f[idx, :, :] = f[idx, :, :] * record_states[idx]
            f[idx, :, :] = f[idx, :, :] * record_states[idx].reshape([node_num, 1])

        # second: update p
        finish_label = True
        for i in range(node_num):
            for j in range(node_num):
                if network_structure[i, j] != 1:
                    assign_value = 0
                else:
                    pos_sum = np.sum(f[:, i, j] * node_positive_record_mask[i, j])
                    if pos_sum + neg_cnt[i, j] != 0:
                        assign_value = pos_sum / (pos_sum + neg_cnt[i, j])
                    else:
                        assign_value = 0

                if finish_label and np.abs(assign_value - p_matrix[i, j]) > max_err:
                    finish_label = False
                p_matrix[i, j] = assign_value
                if assign_value > 1:
                    print("assign value = %f" % assign_value)
                    os.system("pause")

        # third: judge when to stop
        if finish_label:
            # print("\t\tfinish EM process with %d iterations" % iter_cnt)
            break
        # else:
            # print("\t\titerate %d times;" % iter_cnt)

    return p_matrix
