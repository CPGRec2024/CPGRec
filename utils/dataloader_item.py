import os
import sys  
from dgl.data.utils import save_graphs
from tqdm import tqdm
from scipy import stats
from utils.NegativeSampler import NegativeSampler
import pdb
import torch
import logging
logging.basicConfig(stream = sys.stdout, level = logging.INFO)
import numpy as np
import dgl
from dgl.data import DGLDataset
import pandas as pd
from sklearn import preprocessing
from dgl.data import DGLDataset
import pickle


class Dataloader_item_graph(DGLDataset):
    def __init__(self, app_id_path, publisher_path, developer_path, genre_path,dataloader_steam):
        self.app_id_path = app_id_path
        self.publisher_path = publisher_path
        self.developer_path = developer_path
        self.genre_path = genre_path

        logging.info("reading item graph...")
        self.app_id_mapping = dataloader_steam.app_id_mapping    

        self.publisher = dataloader_steam.publisher_mapping     
        self.developer =dataloader_steam.developer_mapping    
        self.genre = dataloader_steam.genre_mapping   

        path_dic_genre = "/CPGRec/data_exist/dic_genre.pkl"
        path_dic_pub = "/CPGRec/data_exist/dic_publisher.pkl"
        path_dic_dev = "/CPGRec/data_exist/dic_developer.pkl"

        if not os.path.exists(path_dic_genre) or not os.path.exists(path_dic_pub) or not os.path.exists(path_dic_dev):
            with open(path_dic_genre, 'wb') as f:
                pickle.dump(self.genre, f)
            with open(path_dic_pub, 'wb') as f:
                pickle.dump(self.publisher, f)
            with open(path_dic_dev, 'wb') as f:
                pickle.dump(self.developer, f)



        path = "CPGRec/data_exist"
        path_graph_and = path + '/graph_and.bin'
        path_graph_or = path + '/graph_or.bin'

        '''graph 1'''
        if os.path.exists(path_graph_and):
            self.graph_and,_ = dgl.load_graphs(path_graph_and)
            self.graph_and = self.graph_and[0]
        else:
            self.genre_pub = self.build_edge_and(self.genre, self.publisher)
            self.genre_dev = self.build_edge_and(self.genre, self.developer)
            self.dev_pub = self.build_edge_and(self.developer, self.publisher)
            
            graph_data_and = {
                ('game', 'co_genre_pub', 'game'): self.genre_pub,
                ('game', 'co_genre_dev', 'game'): self.genre_dev,
                ('game', 'co_dev_pub', 'game'): self.dev_pub
            }
            self.graph_and = dgl.heterograph(graph_data_and)
            dgl.save_graphs(path_graph_and,[self.graph_and])
            


        '''graph 2'''
        if os.path.exists(path_graph_or):
            self.graph_or,_ = dgl.load_graphs(path_graph_or)
            self.graph_or = self.graph_or[0]
        else:
            self.genre_dev_pub = self.build_edge_or(self.genre, self.developer, self.publisher)
            graph_data_or = {
                ('game','co_or','game'): self.genre_dev_pub
            }
            self.graph_or = dgl.heterograph(graph_data_or)
            dgl.save_graphs(path_graph_or,[self.graph_or])


    def build_edge_and(self, mapping1, mapping2):
        src = []
        dst = []
        keys = list(set(mapping1.keys()) & set(mapping2.keys()))

        for game in keys:
            mapping1[game] = set(mapping1[game])
            mapping2[game] = set(mapping2[game])

        for i in range(len(keys) - 1):
            for j in range(i +1, len(keys)):
                game1 = keys[i]
                game2 = keys[j]
                if len(set(mapping1[game1]) & set(mapping1[game2])) > 0 and len(set(mapping2[game1]) & set(mapping2[game2])) > 0:
                    src.extend([game1, game2])
                    dst.extend([game2, game1])
        
        return (torch.tensor(src), torch.tensor(dst))


        
    def build_edge_or(self, mapping1, mapping2, mapping3):
        src = []
        dst = []
        keys = list(set(mapping1.keys()) | set(mapping2.keys()) | set(mapping3.keys()))

        for game in keys:
            if game in mapping1:
                mapping1[game] = set(mapping1[game])
            else:
                mapping1[game] = set()
            if game in mapping2:
                mapping2[game] = set(mapping2[game])
            else:
                mapping2[game] = set()
            if game in mapping3:
                mapping3[game] = set(mapping3[game])
            else:
                mapping3[game] = set()

        for i in range(len(keys) - 1):
            for j in range(i +1, len(keys)): 
                game1 = keys[i]
                game2 = keys[j]
                if len(set(mapping1[game1]) & set(mapping1[game2])) > 0 or len(set(mapping2[game1]) & set(mapping2[game2])) > 0 or len(set(mapping3[game1]) & set(mapping3[game2])) > 0:
                    src.extend([game1, game2])
                    dst.extend([game2, game1])
        
        return (torch.tensor(src), torch.tensor(dst))
