import sys
from util.util import *
from network import construct_network_hsic, net_diffusion_rate_m
from dom_sets import cluster_hier
import time

record_states = load_record_states('./data/record_states.txt')
est_matrix, cov = construct_network_hsic(record_states, sigma=20, permutation_times=1)
p_matrix = net_diffusion_rate_m(record_states, est_matrix)
pc_matrix = stimulate_prob(est_matrix, p_matrix, break_num=1000, save_path=None)
A = cluster_hier(pc_matrix, SIM_MODE="Sim")

