"""
bkdcdd.py

Function to select backdoor graphs and sub-graphs.
"""

import argparse
import os
import sys
from utils.datareader import DataReader
from typing import List, Tuple, Union
sys.path.append('/home/zxx5113/BackdoorGNN/')

import pysnooper
import numpy as np
import copy

AdjMatrix = Union[List[List[int]], np.ndarray]

# return 1D list
def select_cdd_graphs(
        args: argparse.Namespace, data: List[int], adj_list: List[AdjMatrix], subset: str
    ) -> List[int]:
    """ Given a data (train/test), (randomly or determinately) pick up some graph to
    put backdoor trigger, return ids.

    Notes that this function selects candidate graphs based on their sizes, but not
    the original labels. Either the graphs from target class or not have the same
    probability to be selected.

    Arguments
    ---------
    args : argparse.Namespace
        Arguments from command line.

    data : List[int]
        List of graph ids.

    adj_list : List[AdjMatrix]
        List of adjacency matrices (2d-array).

    subset : str
        Subset of data, either 'train' or 'test'.

    Returns
    -------
    List[int]
        List of graph ids.
    """

    if subset not in ('train', 'test'):
        raise ValueError(f'Invalid subset {subset}')

    rs = np.random.RandomState(args.seed)
    graph_sizes = [np.array(adj).shape[0] for adj in adj_list]
    bkd_graph_ratio = args.bkd_gratio_train if subset == 'train' else args.bkd_gratio_test
    bkd_num = int(np.ceil(bkd_graph_ratio * len(data)))

    assert len(data) > bkd_num , "Graph instances are not enough"
    picked_ids = []

    # args.bkd_size: trigger size, default: 5
    # args.bkd_num_pergraph, trigger size, default: 1

    # Randomly pick up graphs as backdoor candidates from data
    remained_set: List[int] = copy.deepcopy(data)   # Candidate pool.
    loopcount = 0
    while bkd_num - len(picked_ids) > 0 and len(remained_set) > 0 and loopcount <= 50:
        loopcount += 1

        # First, pick N graph ids from pool as candidates.
        cdd_ids = rs.choice(remained_set, bkd_num - len(picked_ids), replace=False)

        # Validate if the candidate graph is large enough to inject backdoor.
        for gid in cdd_ids:
            if bkd_num - len(picked_ids) <= 0:
                break

            # Query candidate graph size, if it is too small, skip it.
            gsize = graph_sizes[gid]
            if gsize >= 3 * args.bkd_size * args.bkd_num_pergraph:
                picked_ids.append(gid)

        # Skip picking up small graphs from pool, until there are not enough graphs.
        if len(remained_set) < len(data):
            for gid in cdd_ids:
                if bkd_num - len(picked_ids) <=0:
                    break

                gsize = graph_sizes[gid]
                if gsize >= 1.5 * args.bkd_size * args.bkd_num_pergraph and gid not in picked_ids:
                    picked_ids.append(gid)

        # Skip picking up small graphs from pool, until there are not enough graphs.
        if len(remained_set) < len(data):
            for gid in cdd_ids:
                if bkd_num - len(picked_ids) <= 0:
                    break

                gsize = graph_sizes[gid]
                if gsize >= 1.0 * args.bkd_size * args.bkd_num_pergraph and gid not in picked_ids:
                    picked_ids.append(gid)

        # Conclude all the graphs selected in this round.
        # Use `set()` to avoid duplication.
        picked_ids = list(set(picked_ids))
        remained_set = list(set(remained_set) - set(picked_ids))

        if len(remained_set) == 0 and bkd_num > len(picked_ids):
            print("no more graph to pick, return insufficient candidate graphs, try smaller bkd-pattern or graph size")

    return picked_ids

def select_cdd_nodes(
        args: argparse.Namespace, graph_cdd_ids: List[int], adj_list: List[AdjMatrix]
    ) -> Tuple[List[List[int]], List[List[List[int]]]]:
    """ Given a graph instance, based on pre-determined standard,
    find nodes who should be put backdoor information, return
    their ids.

    Arguments
    ---------
    args : argparse.Namespace
        Arguments from command line.

    graph_cdd_ids : List[int]
        List of graph ids.

    adj_list : List[AdjMatrix]
        List of adjacency matrices (2d-array).

    Returns
    -------
    List[List[int]]
        List of node ids, each list is for a graph.
        bkd nodes under each graph

    List[List[List[int]]]
        List of node groups, each list is for a graph.
        bkd node groups under each graph (in case of each graph has multiple triggers)
    """
    rs = np.random.RandomState(args.seed)

    # step1: find backdoor nodes
    picked_nodes = []  # 2D, save all cdd graphs

    for gid in graph_cdd_ids:
        node_ids = [i for i in range(len(adj_list[gid]))]
        assert len(node_ids) == len(adj_list[gid]), 'node number in graph {} mismatch'.format(gid)

        bkd_node_num = int(args.bkd_num_pergraph * args.bkd_size)
        assert bkd_node_num <= len(adj_list[gid]), "error in SelectCddGraphs, candidate graph too small"

        cur_picked_nodes = rs.choice(node_ids, bkd_node_num, replace=False)
        picked_nodes.append(cur_picked_nodes)

    # step2: match nodes
    assert len(picked_nodes) == len(graph_cdd_ids), "backdoor graphs & node groups mismatch, check SelectCddGraphs/SelectCddNodes"

    node_groups = [] # 3D, grouped trigger nodes
    for i in range(len(graph_cdd_ids)):    # for each graph, divide candidate nodes into groups
        gid = graph_cdd_ids[i]
        nids = picked_nodes[i]

        assert len(nids) % args.bkd_size == 0.0, "Backdoor nodes cannot equally be divided, check SelectCddNodes-STEP1"

        # groups within each graph
        groups = np.array_split(nids, len(nids) // args.bkd_size)
        # np.array_split return list[array([..]), array([...]), ]
        # thus transfer internal np.array into list
        # store groups as a 2D list.
        groups = np.array(groups).tolist()
        node_groups.append(groups)

    assert len(picked_nodes) == len(node_groups), "groups of bkd-nodes mismatch, check SelectCddNodes-STEP2"
    return picked_nodes, node_groups

