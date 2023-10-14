from cal_score import *


class Node:
    """
    me: “我”是哪个节点, int
    parents: 当前已选定的父节点, list
    score：当前父节点下，评分的上界或者评分的确切值, float
    size：问题规模，全体节点个数, int
    next_branch: 下一次分支选择的节点

    """
    def __init__(self, me, parents, score, size, next_branch):
        self.me = me
        self.parents = parents
        self.score = score
        self.size = size
        self.next_branch = next_branch

    def get_children(self, scores):
        """
        :param scores:list, 两个元素，第一个元素为下一次分支选择了某节点的打分上界，
        第二个元素为没有选择某节点的打分上界
        :return:list，两个元素，第一个为选择了某节点的node，第二个为没有选择的node
        """
        children = []
        next_branch = self.next_branch
        if next_branch == self.me:
            next_branch += 1
        if next_branch < self.size:
            child1 = Node(self.me, self.parents+[next_branch], scores[0], self.size, next_branch+1)
            child2 = Node(self.me, self.parents, scores[1], self.size, next_branch+1)
            children.append(child1)
            children.append(child2)
        return children

    def is_leaf_node(self):
        if self.next_branch < self.size-1:
            return False
        elif self.next_branch == self.size-1 and self.me < self.size-1:
            return False
        else:
            return True


def branch_and_bound(data, target_node, score, sigma, permutation_times):
    """
    第一个版本，用score设定最优解，达到剪枝的目的
    """
    best_score = score
    best_parents = []
    history_parents = {}
    history_upper_bound = {}
    root_parents = []
    next_branch = 0
    node_count = 1
    pruning_count = 0
    if target_node == 0:
        next_branch = 1
    root_node = Node(target_node, root_parents, 1, data.shape[1], next_branch)
    expand_set = [root_node]
    while expand_set:
        selected_node = expand_set.pop()
        children = selected_node.get_children([1, 1])
        node_count += 2
        # 计算每一个孩子的上界
        # 第一个孩子，选了某节点
        y_data = data[:, target_node]
        data_size = data.shape[0]
        x_data = np.zeros((data_size, 1))
        for parent in children[0].parents:
            x_data = np.hstack((x_data, np.reshape(data[:, parent], (-1, 1))))
        x_data = x_data[:, 1:]
        upper_bound1 = cal_upper_bound(x_data, y_data, sigma, permutation_times)
        if upper_bound1 >= best_score:
            if children[0].is_leaf_node():
                score1 = cal_score(x_data, y_data, sigma)
                if score1 >= best_score:
                    history_parents[best_score] = best_parents
                    history_upper_bound[best_score] = upper_bound1
                    best_parents = children[0].parents
                    best_score = score1
            else:
                expand_set.append(children[0])
        else:
            pruning_count += 1
            # print("pruning")

        # 第二个孩子， 没有选择某个节点
        y_data = data[:, target_node]
        x_data = np.zeros((data_size, 1))
        for parent in children[1].parents:
            x_data = np.hstack((x_data, np.reshape(data[:, parent], (-1, 1))))
        x_data = x_data[:, 1:]
        upper_bound2 = cal_upper_bound(x_data, y_data, sigma, permutation_times)
        if upper_bound2 >= best_score:
            if children[1].is_leaf_node():
                score2 = cal_score(x_data, y_data, sigma)
                if score2 >= best_score:
                    history_upper_bound[best_score] = upper_bound2
                    history_parents[best_score] = best_parents
                    best_parents = children[1].parents
                    best_score = score2

            else:
                expand_set.append(children[1])
        else:
            pruning_count += 1
            # print("pruning")

    return best_parents, best_score, history_parents, node_count, pruning_count
