import torch.nn as nn
from tqdm import tqdm
import torch
import dgl
import dgl.function as fn
import dgl.nn as dglnn
from dgl.nn.pytorch.conv import GraphConv, GATConv, SAGEConv
import torch.nn.functional as F
from models.submodular_layer import submodular_layer

class Proposed_model(nn.Module):
    def __init__(self, args, graph, graph_and, graph_or, graph_social,device, gamma=80, ablation=False):
        super().__init__()
        self.ablation = ablation
        self.device_ = torch.device(device)
        self.device2 = self.device_
        self.gpu3 = torch.device("cuda:2")
        torch.cuda.empty_cache()
        self.args = args
        self.social_att = args.social_att
        self.param_decay = args.param_decay
        self.GCNor_weight = args.GCNor_weight
        self.hid_dim = args.embed_size  # default = 32
        self.attention_and = args.attention_and
        self.layer_num_and = args.layers_and  
        self.layer_num_or = args.layers_or 
        self.layer_num_khop = args.layers_khop  
        self.layer_num_user_game = args.layers_user_game
        self.graph_and = graph_and.to(self.device_)
        self.graph_or = graph_or.to(self.device_)
        self.graph = graph.to(self.device_)
        self.graph_social = graph_social.to(self.device_)
        self.graph_item2user = dgl.edge_type_subgraph(self.graph,['played by']).to(self.device_)
        self.graph_user2item = dgl.edge_type_subgraph(self.graph,['play']).to(self.device_)
        self.neighbor_num = args.neighbor_num
        self.submodular = args.submodular
        self.popularity = args.popularity
        self.time = args.time
        self.edge_node_weight = True
        

        
        path_weight_edge = "/CPGRec/data_exist/weight_edge.pth"
        path_weight_node = "/CPGRec/data_exist/weight_node.pth"
        self.weight_edge = torch.load(path_weight_edge).to(self.device_)
        self.weight_node = torch.load(path_weight_node).to(self.device_)

        
        

        self.gamma = gamma
        self.w_or = self.gamma / (self.gamma + 2)
        self.w_and = self.w_or / self.gamma
        self.w_self = self.w_or / self.gamma

        self.user_embedding = torch.nn.Parameter(torch.randn(self.graph.nodes('user').shape[0], self.hid_dim)).to(torch.float32)
        self.item_embedding = torch.nn.Parameter(torch.randn(self.graph.nodes('game').shape[0], self.hid_dim)).to(torch.float32)


        self.W_and = torch.nn.Parameter(torch.randn(self.args.embed_size, self.args.embed_size)).to(torch.float32)
        self.a_and = torch.nn.Parameter(torch.randn(self.args.embed_size)).to(torch.float32)
        self.W_or = torch.nn.Parameter(torch.randn(self.args.embed_size, self.args.embed_size)).to(torch.float32)
        self.a_or = torch.nn.Parameter(torch.randn(self.args.embed_size)).to(torch.float32)
        self.W_social = torch.nn.Parameter(torch.randn(self.args.embed_size, self.args.embed_size)).to(torch.float32)
        self.a_social = torch.nn.Parameter(torch.randn(self.args.embed_size)).to(torch.float32)

        self.conv1 = GraphConv(self.hid_dim, self.hid_dim, weight=False, bias=False, allow_zero_in_degree = True).to(self.device_)

        self.build_model_and(self.graph_and)
        self.build_model_or()
        self.build_model_user_game()

    '''attention'''
    def layer_attention(self, ls, W, a):
        tensor_layers = torch.stack(ls, dim=0)
        weight = torch.matmul(tensor_layers, W)
        weight = F.softmax(weight*a, dim=0)
        tensor_layers = torch.sum(tensor_layers * weight, dim=0)
        return tensor_layers




    '''model for graph_and'''

    def build_model_and(self, graph_and):
        self.sub_g1 = dgl.edge_type_subgraph(graph_and,['co_genre_pub']).to(self.device_)
        self.sub_g2 = dgl.edge_type_subgraph(graph_and,['co_genre_dev']).to(self.device_)
        self.sub_g3 = dgl.edge_type_subgraph(graph_and,['co_dev_pub']).to(self.device_)


    def get_h_and(self,attention):
        ls = [self.item_embedding.to(self.device_)]
        h1 = self.item_embedding.to(self.device_)
        h2 = self.item_embedding.to(self.device_)
        h3 = self.item_embedding.to(self.device_)

        for _ in range(self.layer_num_and):
            h1 = self.conv1(self.sub_g1, h1)
            h2 = self.conv1(self.sub_g2, h2)
            h3 = self.conv1(self.sub_g3, h3)
            ls.append(h1)
            ls.append(h2)
            ls.append(h3)

        
        
        if attention == True:
            return self.layer_attention(ls, self.W_and.to(self.device_), self.a_and.to(self.device_))
        else:   
            return ((h1+h2+h3)/3).cpu()



    '''model for graph_or'''

    def build_model_or(self):
        self.model_list_or = nn.ModuleList()
        for _ in range(self.layer_num_or):
            self.model_list_or.append(GraphConv(self.hid_dim, self.hid_dim, weight = False, bias = False, allow_zero_in_degree = True).to(self.device_))


    def get_h_or(self, graph_or):
        ls = [self.item_embedding.to(self.device_)]
        h_temp = self.item_embedding.to(self.device_)
        layer_idx = 1
        param_min = 0.2
        
        for layer in self.model_list_or:
            param = 1-(self.layer_num_or-layer_idx)*self.param_decay
            layer_idx += 1
            param = max(list([param, param_min]))
            h_temp = layer(graph_or, h_temp)
            ls.append(h_temp * param)

        return self.layer_attention(ls, self.W_or.to(self.device_), self.a_or.to(self.device_))




    def build_model_user_game(self):
        self.layers = nn.ModuleList()
        for _ in range(self.layer_num_user_game):
            layer = 0
            if self.edge_node_weight == True:
                layer = GraphConv(self.hid_dim,self.hid_dim, weight=False, bias=False, allow_zero_in_degree = True)
            else:
                layer = dgl.nn.HeteroGraphConv({
                    'play': GraphConv(self.hid_dim,self.hid_dim, weight=False, bias=False, allow_zero_in_degree = True),
                    'played by': GraphConv(self.hid_dim,self.hid_dim, weight=False, bias=False, allow_zero_in_degree = True)
                })
            self.layers.append(layer)
        self.layers.to(self.device_)
    


    '''forward'''
    def forward(self):
        h = {'user':self.user_embedding.to(self.device_), 'game':self.item_embedding.to(self.device_)}
        for layer in self.layers:
            if self.edge_node_weight == True:
                h['game'] = torch.matmul(torch.diag(self.weight_node) , h['game'] )
                h_user = layer(self.graph_item2user, (h['game'],h['user']),edge_weight=self.weight_edge)
                h_item = layer(self.graph_user2item, (h['user'],h['game']))
                h['user'] = h_user
                h['game'] = h_item
            else:
                h = layer(self.graph,h)   


        h_and = self.get_h_and(attention=self.attention_and)
        h_or = self.get_h_or(self.graph_or)
        h_self = h['game']

        h['game'] = self.w_and * h_and + self.w_or * h_or  + self.w_self * h_self
        

        return h
