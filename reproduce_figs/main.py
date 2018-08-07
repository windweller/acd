import numpy as np
# import matplotlib.pyplot as plt
import torch
import pickle
import sys
from torch.autograd import Variable

sys.path.append('..')
sys.path.append('../visualization')
sys.path.append('../acd/util')
sys.path.append('../acd/scores')
sys.path.append('../acd/agglomeration')
# import viz_1d as viz
import tiling_1d as tiling
import agg_1d as agg
import cd
import score_funcs


# form class to hold data
class B:
    def __init__(self):
        self.text = Variable(torch.zeros(1).cuda())

sys.path.append('../dsets/sst')
import dset
sst_pkl = pickle.load(open('../dsets/sst/sst.pkl', 'rb'))

snapshot_file = '../dsets/sst/sst.model'
model = dset.get_model(snapshot_file).eval()

### ZYH
embed = model.embed.weight.data.cpu().numpy() 
unk_embed = np.reshape(embed[0], (1, len(embed[0])))
embed = np.concatenate([embed, unk_embed], 0)
x, y = embed.shape
model.embed = torch.nn.Embedding(x, y).cuda()
model.embed.weight.data.copy_(torch.from_numpy(embed))
### ZYH

# base parameters
sweep_dim = 1 # how large chunks of text should be considered (1 for words)
method = 'cd' # build_up, break_down, cd
percentile_include = 99.5 # keep this very high so we don't add too many words at once
num_iters = 25 # maximum number of iterations (rarely reached)

# text and label
sentence = ['it', "'s", 'a', 'lovely', 'film', 'with', 'lovely', 'performances', 'by', 'Buy', 'and', 'Accorsi', '.']
sentence = ['Yet', 'the', 'act', 'is', 'still', 'charming', 'here', '.']
# sentence = ['a', 'great', 'ensemble', 'cast', 'ca', 'n\'t', 'lift', 'this', 'heartfelt', 'enterprise', 'out', 'of', 'the', 'familiar', '.'] 
label = 0 # 0 if positive 1 if negative


def batch_from_str_list(s):
    batch = B()
    nums = np.expand_dims(np.array([sst_pkl['stoi'].get(x, len(sst_pkl['stoi'])) for x in s]).transpose(), axis=1)
    batch.text.data = torch.LongTensor(nums).cuda()
    return batch

# prepare inputs
batch = batch_from_str_list(sentence)
print(batch.text.data)
scores_all = model(batch).data.cpu().numpy()[0] # predict
print(scores_all)
label_pred = np.argmax(scores_all) # get predicted class

# agglomerate
lists = agg.agglomerate(model, batch, percentile_include, method, sweep_dim, # only works for sweep_dim = 1
                    sentence, label_pred, num_iters=num_iters) # see agg_1d.agglomerate to understand what this dictionary contains
lists = agg.collapse_tree(lists) # don't show redundant joins
print(lists)
# visualize
# viz.word_heatmap(sentence, lists, label_pred, label, fontsize=9)