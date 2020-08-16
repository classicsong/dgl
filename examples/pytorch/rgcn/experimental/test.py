import argparse
import os
os.environ['DGLBACKEND']='pytorch'

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
import dgl
from dgl import DGLGraph

def init_emb(shape, dtype):
    arr = th.ones(shape, dtype=dtype)
    return arr

if __name__ == '__main__':
    dgl.distributed.initialize('172.0.0.1', 1, num_workers=1)
    g = dgl.distributed.DistGraph('172.0.0.1', 'ogbn-mag', part_config='data/ogbn-mag.json')
    embed_size=10

    w1 = nn.Parameter(th.Tensor(10, 10))
    w2 = nn.Parameter(th.Tensor(10, 10))
    nn.init.xavier_uniform_(w1)
    w2.data[:] = w1.data

    print(w1.data)
    print(w2.data)

    dgl_emb = dgl.distributed.DistEmbedding(
            g.number_of_nodes(),
            embed_size,
            'test',
            init_emb)
    torch_embeds = th.nn.Embedding(g.number_of_nodes(), embed_size, sparse=True)
    nn.init.ones_(torch_embeds.weight)

    emb_optimizer = dgl.distributed.SparseAdagrad([dgl_emb], lr=0.01)
    th_emb_optimizer = th.optim.SparseAdam(torch_embeds.parameters(), lr=0.01)

    idx = th.arange(10).long()
    idx2 = th.zeros((10,)).long()
    print(idx.shape)
    print(idx2.shape)
    idx = th.cat([idx, idx2])

    truth = th.ones((20,)).long()

    th_emb_optimizer.zero_grad()
    dgl_res = dgl_emb(idx)
    th_res = torch_embeds(idx)
    print(dgl_res)
    print(th_res)
    result1 = dgl_res @ w1
    result2 = th_res @ w2

    loss1 = F.cross_entropy(result1, truth)
    loss2 = F.cross_entropy(result2, truth)
    loss1.backward()
    loss2.backward()
    emb_optimizer.step()
    th_emb_optimizer.step()

    print(torch_embeds.weight.grad)
 
    print(dgl_emb(idx))
    print(torch_embeds(idx))
