"""
This uses original ACD code to generate a tree
extremely slow
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import pickle
import sys
from torch.autograd import Variable
from nltk import Tree

sys.path.append('..')
sys.path.append('../visualization')
sys.path.append('../acd/util')
sys.path.append('../acd/scores')
sys.path.append('../acd/agglomeration')
import viz_1d as viz
import tiling_1d as tiling
import agg_1d as agg
import cd
import score_funcs

from utils import chomsky_normal_form


# form class to hold data
class B:
    def __init__(self):
        self.text = Variable(torch.zeros(1))


sys.path.append('../dsets/sst')
import dset

sst_pkl = pickle.load(open('../dsets/sst/sst.pkl', 'rb'))

# snapshot_file = '../dsets/sst/sst.model'
snapshot_file = '../dsets/sst/results/best_snapshot_devacc_84_devloss_0.441337406635__iter_4000_model.pt'
model = dset.get_model(snapshot_file, cpu=True).eval()
model.use_gpu = False

embed = model.embed.weight.data.numpy()
unk_embed = np.reshape(embed[0], (1, len(embed[0])))
embed = np.concatenate([embed, unk_embed], 0)
x, y = embed.shape
model.embed = torch.nn.Embedding(x, y)
model.embed.weight.data.copy_(torch.from_numpy(embed))


# [l, r], a = previous line, b = current line
# children: [num_words]
# tree_dict: {}
def build_tree(l, r, a, b, score, tree_dict):
    ret = []
    i = l
    while i <= r:
        if a[i] != 0:
            j = nonzero(i, a)
            ret.append(tree_dict[(i, j)])
            i = j + 1
        else:
            ret.append(tree_dict[i])
            i = i + 1
    ret = Tree("{:.2f}".format(score), ret)
    return ret


def word_score_list_to_comp_score_list(orig_word_socres_list):
    # turn numpy array [word-score-1, word-score-2, ..]
    # into dictionary {1: word-score-1, 2: word-score-2, ...}
    comp_scores_list = {}
    for i in range(0, orig_word_socres_list.size):
        comp_scores_list[i] = orig_word_socres_list[i]
    return comp_scores_list


def build_leaf_nodes(word_socres_list, sent):
    nodes = []
    for i in range(len(word_socres_list)):
        nodes.append(Tree("{:.2f}".format(word_socres_list[i]), [sent[i]]))
    return nodes


def tree_to_str(tree):
    parse_string = ' '.join(str(tree).split())
    return parse_string


from copy import copy, deepcopy


def fill_comp_list_w_previous(lists, num_words, filler_index=-1):
    # comps_list only record change (probably collapsed)

    num_iters = len(lists['comps_list'])
    new_comp_scores_list = [lists['comps_list'][0]]

    prev_list = None
    for i in range(1, num_iters):
        cur_list = copy(lists['comps_list'][i])
        if prev_list is None:
            new_comp_scores_list.append(cur_list)
        else:
            for j in range(0, num_words):
                if prev_list[j] != 0 and cur_list[j] == 0:
                    cur_list[j] = filler_index
            new_comp_scores_list.append(cur_list)

        prev_list = cur_list

    return new_comp_scores_list


# Input: (0, 2, [1,1,1,1,1,0,1])
# Output: (0, 4)
# Input: (0, 1, [1,1,1,1,1,1,1])
# Output: (0, 6)
def expand(l, r, b):
    i, j = l, r
    while i >= 0 and b[i] != 0:
        i -= 1
    i += 1
    while j < len(b) and b[j] != 0:
        j += 1
    j -= 1
    return i, j


# Input: (2, [0,1,1,1,0])
# Output: 3
# Input: (2, [0,1,1,1])
# Output: 3
def nonzero(l, a):
    r = l
    while r < len(a) and a[r] != 0:
        r += 1
    r -= 1
    return r


# Input: [1,1,0,0,2,2,0,4,4,4,0,0]
# Output: [(0, 1), (4, 5), (7, 9)]
def get_lrlist(a):
    ret = []
    i = 0
    while i < len(a):
        while i < len(a) and a[i] == 0:
            i += 1
        if i == len(a): break
        j = i
        while j < len(a) and a[j] != 0:
            j += 1
        j -= 1
        ret.append((i, j))
        i = j + 1
    return ret


def comp_to_tree(lists, sent, binarize=True):
    # return a parentheses based tree representation
    # NLTK will transform it into a binary tree. We go with left-branching for non-binary nodes
    # sent: a list of words
    # binarize=True: we use left binarization
    # main logic: we replicate the tree to fill the position of children!

    word_socres_list = word_score_list_to_comp_score_list(lists['scores_list'][0])

    children = build_leaf_nodes(word_socres_list, sent)

    # [tree(root), ..., tree(root)]
    tree_dict = {}
    for c_i, c in enumerate(children):
        tree_dict[c_i] = c

    num_iters = len(lists['comps_list'])
    num_words = len(sent)

    comps_list = fill_comp_list_w_previous(lists, num_words)

    prev_level = None
    for i in range(1, num_iters):
        comps = comps_list[i]  # lists['comps_list'][i]
        comp_scores = lists['comp_scores_list'][i]

        if prev_level is None:
            # [2 3 5], we ignore 0
            u_values = np.unique(comps[comps.nonzero()])
            # We build parent node, and then replace the children node
            # [0 1 2 3 ... 14] -> [0 t1 t1 3 ... 14]
            for v in u_values:
                # these two children are merged!
                child_indices = np.nonzero(comps == v)[0]
                # print("{}: {}".format(v, child_indices))
                parent_val = comp_scores[v]
                parent = Tree("{:.2f}".format(parent_val), [tree_dict[j] for j in child_indices])

                l, r = min(child_indices), max(child_indices)
                tree_dict[(l, r)] = parent

            prev_level = comps
        else:
            # a is prev_level
            # b comps
            tree_components = []

            # u_values = np.unique(prev_level[prev_level.nonzero()])
            # TEST THIS
            # print(prev_level)
            a = get_lrlist(prev_level)
            # print(a)
            r_last = -1
            for tup in a:
                l, r = tup
                if l < r_last:
                    continue
                l_, r_ = expand(l, r, comps)
                r_last = r
                # print(l, r, l_, r_, comps)
                # tree_components.append()
                if (l_, r_) in tree_dict:
                    continue
                parent_val = comp_scores[comps[l_]]
                t = build_tree(l_, r_, prev_level, comps, parent_val, tree_dict)
                tree_dict[(l_, r_)] = t

            prev_level = comps

    return [tree_dict[(0, num_words - 1)]]


def get_sentences(path):
    sents = []
    with open(path, 'r') as f:
        for line in f:
            sent = Tree.fromstring(line.strip()).leaves()
            sents.append(sent)
    return sents

from nltk import treetransforms
import time
import json

def batch_from_str_list(s):
    batch = B()
    nums = np.expand_dims(np.array([sst_pkl['stoi'].get(x, len(sst_pkl['stoi'])) for x in s]).transpose(), axis=1)
    batch.text.data = torch.LongTensor(nums)
    return batch

def transform_sst_to_acd_trees(sents, tag='train', filter_len=20):
    # base parameters
    sweep_dim = 1  # how large chunks of text should be considered (1 for words)
    method = 'cd'  # build_up, break_down, cd
    percentile_include = 99.5  # keep this very high so we don't add too many words at once
    num_iters = 25  # maximum number of iterations (rarely reached)

    new_data = []
    cnt = time.time()

    for s_i, sent in enumerate(sents):
        # prepare inputs
        # print(len(sent))
        if s_i % 10 == 0:
            print("phase {} time {} processed {}".format(tag, time.time() - cnt, s_i))

        if len(sent) > filter_len:
            continue
        # print("{} time {} - 0".format(s_i, time.time() - cnt))
        # cnt = time.time()

        sent = [w.lower() for w in sent]

        batch = batch_from_str_list(sent)
        scores_all = model(batch).data.numpy()[0]  # predict
        label_pred = np.argmax(scores_all)  # get predicted class

        # agglomerate
        # print("{} time {} - 1".format(s_i, time.time() - cnt))
        # cnt = time.time()
        lists = agg.agglomerate(model, batch, percentile_include, method, sweep_dim,  # only works for sweep_dim = 1
                                sent, label_pred,
                                num_iters=num_iters)  # see agg_1d.agglomerate to understand what this dictionary contains
        # print("{} time {} - 1.5".format(s_i, time.time() - cnt))
        lists = agg.collapse_tree(lists)  # don't show redundant joins

        # print("{} time {} - 2".format(s_i, time.time() - cnt))
        # cnt = time.time()
        # gather tree
        children = comp_to_tree(lists, sent)

        # print("{} time {} - 3".format(s_i, time.time() - cnt))
        # cnt = time.time()
        # uniary combine the tree, then binarize it
        tree = deepcopy(children[0])
        treetransforms.collapse_unary(tree)

        chomsky_normal_form(tree, factor='left')

        new_data.append(tree_to_str(tree))

        if s_i % 100 == 0:
            json.dump(new_data, open('tmp.json', 'w'))

    return new_data

import IPython

if __name__ == '__main__':
    sst_path = "../data/sst/trees/"
    train_path = sst_path + "train.txt"
    dev_path = sst_path + "dev.txt"
    test_path = sst_path + "test.txt"

    train_sents = get_sentences(train_path)
    dev_sents = get_sentences(dev_path)
    test_sents = get_sentences(test_path)

    new_dev_sents = transform_sst_to_acd_trees(dev_sents, 'dev')
    new_test_sents = transform_sst_to_acd_trees(test_sents, 'test')
    new_train_sents = transform_sst_to_acd_trees(train_sents)

    print(len(new_train_sents))
    print(len(new_dev_sents))
    print(len(new_test_sents))

    with open('../data/sst/acd_trees/train.txt', 'w') as f:
        for s in new_train_sents:
            f.write(s + '\n')

    with open('../data/sst/acd_trees/dev.txt', 'w') as f:
        for s in new_dev_sents:
            f.write(s + '\n')

    with open('../data/sst/acd_trees/test.txt', 'w') as f:
        for s in new_test_sents:
            f.write(s + '\n')

    IPython.embed()  # in the end we keep everything