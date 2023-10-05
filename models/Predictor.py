import torch.nn as nn
import dgl.function as fn
import torch
class Predictor(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():

            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype = etype)
            return graph.edges[etype].data['score']
