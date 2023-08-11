import argparse
import sys, os
import pysnooper
sys.path.append(os.path.abspath('..'))

import copy
import numpy as np
from typing import List, Tuple, Union
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from utils.datareader import DataReader
from utils.bkdcdd import select_cdd_graphs, select_cdd_nodes
from utils.mask import gen_mask, recover_mask
import main.benign as benign
import trojan.GTA as gta
from trojan.input import gen_input
from trojan.prop import train_model, evaluate
from config import parse_args

class GraphBackdoor:
    benign_dr: DataReader
    benign_model: nn.Module

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

        assert torch.cuda.is_available(), 'no GPU available'

        self.cpu = torch.device('cpu')
        self.cuda = torch.device('cuda')
        self.benign_dr = None
        self.benign_model = None

    @pysnooper.snoop()
    def run(self):
        # train a benign GNN
        self.benign_dr, self.benign_model = benign.run(self.args)
        model = copy.deepcopy(self.benign_model).to(self.cuda)

        # pick up initial candidates
        bkd_gids_test, _, bkd_nid_groups_test = self.bkd_cdd('test')

        # figure out the max node number and feature dimension in entire dataset
        # this is used for constructing the trigger generator,
        # which is a fully connected network.
        nodenums = [adj.shape[0] for adj in self.benign_dr.data['adj_list']]
        nodemax = max(nodenums)
        featdim = np.array(self.benign_dr.data['features'][0]).shape[1]

        # init two generators for topology / feature
        toponet = gta.GraphTrojanNet(nodemax, self.args.gtn_layernum)
        featnet = gta.GraphTrojanNet(featdim, self.args.gtn_layernum)

        # init test data, fill values in adjacency matrix and features as 0
        # NOTE: for data that can only add perturbation on features, only init the topology value
        init_dr_test = self.init_trigger(
            self.args, copy.deepcopy(self.benign_dr), bkd_gids_test, bkd_nid_groups_test, 0.0, 0.0)
        bkd_dr_test = copy.deepcopy(init_dr_test)

        # create mask, such that the benign nodes are not modified.
        topomask_test, featmask_test = gen_mask(
            init_dr_test, bkd_gids_test, bkd_nid_groups_test)

        # create input for the trigger generator.
        Ainput_test, Xinput_test = gen_input(self.args, init_dr_test, bkd_gids_test)

        for rs_step in range(self.args.resample_steps):   # for each step, choose different sample
            # randomly select new graph backdoor samples
            bkd_gids_train, _, bkd_nid_groups_train = self.bkd_cdd('train')

            # positive / negative sample set
            pset = bkd_gids_train
            nset = list(set(self.benign_dr.data['splits']['train']) - set(pset))

            # pn_rate: the ratio between trigger-embedded graphs (positive) and
            #          benign ones (negative)
            # default: 1.0
            if self.args.pn_rate != None:
                if len(pset) > len(nset):
                    repeat = int(np.ceil(len(pset)/(len(nset)*self.args.pn_rate)))
                    nset = list(nset) * repeat
                else:
                    repeat = int(np.ceil((len(nset)*self.args.pn_rate)/len(pset)))
                    pset = list(pset) * repeat

            init_dr_train = self.init_trigger(
                self.args, copy.deepcopy(self.benign_dr), bkd_gids_train, bkd_nid_groups_train, 0.0, 0.0)
            bkd_dr_train = copy.deepcopy(init_dr_train)

            topomask_train, featmask_train = gen_mask(
                init_dr_train, bkd_gids_train, bkd_nid_groups_train)
            Ainput_train, Xinput_train = gen_input(self.args, init_dr_train, bkd_gids_train)

            for bi_step in range(self.args.bilevel_steps):
                print("Resampling step %d, bi-level optimization step %d" % (rs_step, bi_step))

                toponet, featnet = gta.train_gtn(
                    self.args, model, toponet, featnet,
                    pset, nset, topomask_train, featmask_train,
                    init_dr_train, bkd_dr_train, Ainput_train, Xinput_train)

                # get new backdoor datareader for training based on well-trained generators
                for gid in bkd_gids_train:
                    rst_bkdA = toponet(
                        Ainput_train[gid], topomask_train[gid], self.args.topo_thrd,
                        self.cpu, self.args.topo_activation, 'topo')
                    # rst_bkdA = recover_mask(nodenums[gid], topomask_train[gid], 'topo')
                    # bkd_dr_train.data['adj_list'][gid] = torch.add(rst_bkdA, init_dr_train.data['adj_list'][gid])
                    bkd_dr_train.data['adj_list'][gid] = torch.add(
                        rst_bkdA[:nodenums[gid], :nodenums[gid]].detach().cpu(),
                        init_dr_train.data['adj_list'][gid])

                    rst_bkdX = featnet(
                        Xinput_train[gid], featmask_train[gid], self.args.feat_thrd,
                        self.cpu, self.args.feat_activation, 'feat')
                    # rst_bkdX = recover_mask(nodenums[gid], featmask_train[gid], 'feat')
                    # bkd_dr_train.data['features'][gid] = torch.add(rst_bkdX, init_dr_train.data['features'][gid])
                    bkd_dr_train.data['features'][gid] = torch.add(
                        rst_bkdX[:nodenums[gid]].detach().cpu(), init_dr_train.data['features'][gid])

                # train GNN
                train_model(self.args, bkd_dr_train, model, list(set(pset)), list(set(nset)))

                #----------------- Evaluation -----------------#
                for gid in bkd_gids_test:
                    rst_bkdA = toponet(
                        Ainput_test[gid], topomask_test[gid], self.args.topo_thrd,
                        self.cpu, self.args.topo_activation, 'topo')
                    # rst_bkdA = recover_mask(nodenums[gid], topomask_test[gid], 'topo')
                    # bkd_dr_test.data['adj_list'][gid] = torch.add(rst_bkdA,
                    #     torch.as_tensor(copy.deepcopy(init_dr_test.data['adj_list'][gid])))
                    bkd_dr_test.data['adj_list'][gid] = torch.add(
                        rst_bkdA[:nodenums[gid], :nodenums[gid]],
                        torch.as_tensor(copy.deepcopy(init_dr_test.data['adj_list'][gid])))

                    rst_bkdX = featnet(
                        Xinput_test[gid], featmask_test[gid], self.args.feat_thrd,
                        self.cpu, self.args.feat_activation, 'feat')
                    # rst_bkdX = recover_mask(nodenums[gid], featmask_test[gid], 'feat')
                    # bkd_dr_test.data['features'][gid] = torch.add(
                    #     rst_bkdX, torch.as_tensor(copy.deepcopy(init_dr_test.data['features'][gid])))
                    bkd_dr_test.data['features'][gid] = torch.add(
                        rst_bkdX[:nodenums[gid]], torch.as_tensor(copy.deepcopy(init_dr_test.data['features'][gid])))

                # graph originally in target label
                yt_gids = [gid for gid in bkd_gids_test
                        if self.benign_dr.data['labels'][gid]==self.args.target_class]
                # graph originally notin target label
                yx_gids = list(set(bkd_gids_test) - set(yt_gids))
                clean_graphs_test = list(set(self.benign_dr.data['splits']['test'])-set(bkd_gids_test))

                # feed into GNN, test success rate
                bkd_acc = evaluate(self.args, bkd_dr_test, model, bkd_gids_test)
                flip_rate = evaluate(self.args, bkd_dr_test, model,yx_gids)
                clean_acc = evaluate(self.args, bkd_dr_test, model, clean_graphs_test)

                # save gnn
                if rs_step == 0 and (bi_step==self.args.bilevel_steps-1 or abs(bkd_acc-100) <1e-4):
                    if self.args.save_bkd_model:
                        save_path = self.args.bkd_model_save_path
                        os.makedirs(save_path, exist_ok=True)
                        save_path = os.path.join(save_path, '%s-%s-%f.t7' % (
                            self.args.model, self.args.dataset, self.args.train_ratio,
                            self.args.bkd_gratio_trainset, self.args.bkd_num_pergraph, self.args.bkd_size))

                        torch.save({'model': model.state_dict(),
                                    'asr': bkd_acc,
                                    'flip_rate': flip_rate,
                                    'clean_acc': clean_acc,
                                }, save_path)
                        print("Trojaning model is saved at: ", save_path)

                if abs(bkd_acc-100) <1e-4:
                    # bkd_dr_tosave = copy.deepcopy(bkd_dr_test)
                    print("Early Termination for 100% Attack Rate")
                    break
        print('Done')


    def bkd_cdd(
            self, subset: str
        ) -> Tuple[List, List, List]:
        """ Select candidate graphs and nodes for backdoor injection.

        Arguments
        ---------
        subset : str
            target split to inject backdoor. Should be 'train' or 'test'.

        Returns
        -------
        List[int]
            List of graph ids to be injected backdoor.

        List[List[int]]
            List of node ids to be injected backdoor.

        List[List[List[int]]]
            List of node groups to be injected backdoor.
        """
        if subset not in ('train', 'test'):
            raise ValueError(f'Invalid subset: {subset}')

        # pick graphs to add trigger (not modify now) => graph ids
        bkd_gids = select_cdd_graphs(
            self.args, self.benign_dr.data['splits'][subset], self.benign_dr.data['adj_list'], subset)

        # find trigger nodes (sub-graph) per graph
        # same sequence with selected backdoored graphs
        bkd_nids, bkd_nid_groups = select_cdd_nodes(
            self.args, bkd_gids, self.benign_dr.data['adj_list'])

        # Sanity check, otherwise its an internal error
        assert len(bkd_gids) == len(bkd_nids) == len(bkd_nid_groups)

        return bkd_gids, bkd_nids, bkd_nid_groups


    @staticmethod
    def init_trigger(
            args, dr: DataReader, bkd_gids: List[int], bkd_nid_groups: List[List[List[int]]],
            init_edge: float, init_feat: float
        ):
        """ Initialize the trigger in the target data.

        Arguments
        ---------
        args : argparse.Namespace
            Arguments from command line.

        dr : DataReader
            DataReader object.

        bkd_gids : List[int]
            List of graph ids to be injected backdoor.

        bkd_nid_groups : List[List[List[int]]]
            List of node groups to be injected backdoor.

        init_edge : float
            Initial value of the trigger edge.

        init_feat : float
            Initial value of the trigger feature.
        """
        if args.target_class == None:
            raise ValueError('Target class is not specified')

        if init_feat == None:
            init_feat = -1
            print('init feat == None, transferred into -1')

        # (in-place) datareader trigger injection
        for i in tqdm(range(len(bkd_gids)), desc="initializing trigger...", ncols=0):
            gid = bkd_gids[i]
            for group in bkd_nid_groups[i] :
                # construct adjacency list (format: src->dst)
                src, dst = [], []
                for v1 in group:
                    for v2 in group:
                        if v1!=v2:
                            src.append(v1)
                            dst.append(v2)

                # change adjacency matrix in-place
                a = np.array(dr.data['adj_list'][gid])  # Query the original adjacency matrix
                a[src, dst] = init_edge                 # Call np slice to modify the matrix element
                dr.data['adj_list'][gid] = a.tolist()   # Write back to the datareader object

                # change features in-place
                featdim = len(dr.data['features'][0][0])
                a = np.array(dr.data['features'][gid])
                a[group] = np.ones((len(group), featdim)) * init_feat
                dr.data['features'][gid] = a.tolist()

            # change graph labels, use dirty label attack to boost attack success rate
            dr.data['labels'][gid] = args.target_class

        return dr

if __name__ == '__main__':
    args = parse_args()
    attack = GraphBackdoor(args)
    attack.run()