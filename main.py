import sys
sys.path.append('../')
import dgl
import dgl.function as fn
import os
import multiprocessing as mp
from tqdm import tqdm
import pdb
import random
import numpy as np
import torch
import torch.nn as nn
import logging
logging.basicConfig(stream = sys.stdout, level = logging.INFO)
from utils.parser import parse_args

from utils.dataloader_steam import Dataloader_steam_filtered

from utils.dataloader_item import Dataloader_item_graph
from models.model import Proposed_model
from models.Predictor import Predictor
import pickle


from utils.Get_Weight import get_weight

game_cold = set(torch.load("/CPGRec/data_exist/game_cold.pth")[0].tolist())
ls_5 = []
ls_10 = []
ls_20 = []


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def get_valid_mask(DataLoader, graph, valid_user):
    path = '/CPGRec/data_exist'
    path_valid_mask_trail = path+"/valid_mask.pth"
    if os.path.exists(path_valid_mask_trail):
        valid_mask = torch.load(path_valid_mask_trail)
        return valid_mask
    else:
        valid_mask = torch.zeros(len(valid_user), graph.num_nodes('game'))
        for i in range(len(valid_user)):
            user = valid_user[i]
            item_train = torch.tensor(DataLoader.dic_user_game[user])
            valid_mask[i, :][item_train] = 1
        valid_mask = valid_mask.bool()
        torch.save(valid_mask, path_valid_mask_trail)
        return valid_mask



def construct_negative_graph(graph, etype,device):

    utype, _ , vtype = etype
    src, _ = graph.edges(etype = etype)
    src = src.to(device)
    dst = torch.randint(graph.num_nodes(vtype), size = src.shape).to(device)
    return dst, dgl.heterograph({etype: (src, dst)}, num_nodes_dict = {ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})


def get_coverage(ls_tensor, mapping1,mapping2,mapping3):
    set_1 = set()
    set_2 = set()
    set_3 = set()
    
    for i in ls_tensor:
        if int(i) in mapping1.keys():
            type_1 = mapping1[int(i)]
            set_1 = set_1.union(set(type_1))

        if int(i) in mapping2.keys():
            type_2 = mapping2[int(i)]
            set_2 = set_2.union(set(type_2))

        if int(i) in mapping3.keys():
            type_3 = mapping3[int(i)]
            set_3 = set_3.union(set(type_3))

    return float(len(set_1)+len(set_2)+len(set_3))





def validate(valid_mask, dic, h, ls_k, mapping1,mapping2,mapping3, to_get_coverage):

    users = torch.tensor(list(dic.keys())).long()
    user_embedding = h['user'][users]
    game_embedding = h['game']
    rating = torch.mm(user_embedding, game_embedding.t())
    rating[valid_mask] = -float('inf')

    valid_mask = torch.zeros_like(valid_mask)
    for i in range(users.shape[0]):
        user = int(users[i])
        items = torch.tensor(dic[user])
        valid_mask[i, items] = 1

    _, indices = torch.sort(rating, descending = True)
    ls = [valid_mask[i,:][indices[i, :]] for i in range(valid_mask.shape[0])]
    result = torch.stack(ls).float()

    res = []
    ndcg = 0
    for k in ls_k:

        discount = (torch.tensor([i for i in range(k)]) + 2).log2()
        ideal, _ = result.sort(descending = True)
        idcg = (ideal[:, :k] / discount).sum(dim = 1)
        dcg = (result[:, :k] / discount).sum(dim = 1)
        ndcg = torch.mean(dcg / idcg)


        recall = torch.mean(result[:, :k].sum(1) / result.sum(1))
        hit = torch.mean((result[:, :k].sum(1) > 0).float())
        precision = torch.mean(result[:, :k].mean(1))


        if to_get_coverage == False:
            coverage = -1
        else:
            cover_tensor = torch.tensor([get_coverage(indices[i,:k],mapping1,mapping2,mapping3) for i in range(users.shape[0])])
            coverage = torch.mean(cover_tensor)


        logging_result = "For k = {}, ndcg = {}, recall = {}, hit = {}, precision = {}, coverage = {}".format(k, ndcg, recall, hit, precision, coverage)
        logging.info(logging_result)
        res.append(logging_result)
    return  coverage, str(res)



if __name__ == '__main__':

    seed = int(input("seed: "))
    setup_seed(seed)
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu}")

    path = "/CPGRec/steam_data"

    user_id_path = path + '/users.txt'
    app_id_path = path + '/app_id.txt'
    app_info_path = path + '/App_ID_Info.txt'
    friends_path = path + '/friends.txt'
    developer_path = path + '/Games_Developers.txt'
    publisher_path = path + '/Games_Publishers.txt'
    genres_path = path + '/Games_Genres.txt'

    

    DataLoader = Dataloader_steam_filtered(args, path, user_id_path, app_id_path, app_info_path, friends_path, developer_path, publisher_path, genres_path)
    graph = DataLoader.graph.to(device)
    DataLoader_item = Dataloader_item_graph( app_id_path, publisher_path, developer_path, genres_path, DataLoader)
    graph_item_and = DataLoader_item.graph_and
    graph_item_or = DataLoader_item.graph_or
    graph_social = dgl.edge_type_subgraph(graph, [('user','friend of','user')])
    graph = dgl.edge_type_subgraph(graph, [('user','play','game'),('game','played by','user')])
    

    
    valid_user = list(DataLoader.valid_data.keys())
    valid_mask = get_valid_mask(DataLoader, graph, valid_user)


    h_weight = graph.edata['time'][('game','played by','user')]
    h_weight += 0.5
    graph.edata['time'][('game','played by','user')] = h_weight

    
    model = Proposed_model(args, graph, graph_item_and, graph_item_or,
                        graph_social,device, gamma=args.gamma, ablation = False)
    model.to(device)    



    predictor = Predictor()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    stop_count = 0
    ls_k = args.k
    total_epoch = 0
    
    loss_pre = 0
    loss = 0
    test_result = 0
    coverage = 0


    for epoch in range(args.epoch):
        print("="*40)
        model.train()
        dst, graph_neg = construct_negative_graph(graph,('user','play','game'),device)
        h = model()

        score = predictor(graph, h, ('user','play','game'),device)
        score_neg = predictor(graph_neg, h, ('user','play','game'),device)
        loss_pre = loss



        score_neg_reweight = score_neg * (score_neg.sigmoid()*args.K)
        loss =  (-((score - score_neg_reweight).sigmoid().clamp(min=1e-8, max=1-1e-8).log())).sum()

        loss = loss.to(device)
        logging.info('Epoch {}'.format(epoch))
        logging.info(f"loss = {loss}\n")
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_epoch += 1
        
        to_get_coverage = False

        if total_epoch > 1:
            model.eval()
            logging.info("begin validation")

            _, result = validate(valid_mask, DataLoader.valid_data, h, ls_k, DataLoader_item.genre,DataLoader_item.developer,DataLoader_item.publisher, to_get_coverage)
            logging.info(result)


            if loss < loss_pre:
                stop_count = 0
                logging.info("begin test")
                _, test_result = validate(valid_mask, DataLoader.test_data, h, ls_k, DataLoader_item.genre,DataLoader_item.developer,DataLoader_item.publisher, to_get_coverage)
                logging.info(test_result)
            else:
                stop_count += 1
                logging.info(f"stop count:{stop_count}")
                if stop_count > args.early_stop:
                    logging.info('early stop')
                    break

    
    logging.info(test_result)

    torch.save(model, path_model)

