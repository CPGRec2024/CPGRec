import os
import sys
from dgl.data.utils import save_graphs
from tqdm import tqdm
from scipy import stats
import pdb
import torch
import logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import numpy as np
import dgl
from dgl.data import DGLDataset
import pandas as pd
from sklearn import preprocessing
import pickle
from utils.distribution_of_gametime import get_sigmoid_score
import dgl.function as fn
import pandas as pd



class Dataloader_steam_filtered(DGLDataset):
    def __init__(self, args, path, user_id_path, app_id_path, app_info_path, friends_path, developer_path, genre_path, publisher_path,
                 device = 'cpu', name = 'steam'):
        
        logging.info("steam dataloader init...")

        self.args = args
        self.path = path
        self.user_id_path = self.path+"/users.txt"
        self.app_info_path = self.path+"/App_ID_Info.txt"
        
        self.app_id_path = self.path+"/app_id.txt"
        self.friends_path = self.path+"/friends.txt"
        self.genre_path = self.path+"/Games_Genres.txt"
        self.developer_path = self.path+"/Games_Developers.txt"
        self.publisher_path = self.path+"/Games_Publishers.txt"
        self.train_game_path = self.path+"/train_game.txt"
        self.valid_game_path = self.path+"/valid_data/valid_game.txt"
        self.test_game_path = self.path+"/test_data/test_game.txt"
        self.train_time_path = self.path+"/train_time.txt"

        self.graph_path = self.path + "/graph.bin"
        
        '''get user id mapping and app id mapping'''
        logging.info("reading user id mapping:")
        self.user_id_mapping = self.read_user_id_mapping(self.user_id_path)
        logging.info("reading app id mapping:")
        self.app_id_mapping = self.read_app_id_mapping(self.app_id_path)
        
        


        '''build valid and test data'''

        logging.info("build valid data:")
        self.valid_data = self.build_valid_data(self.valid_game_path)
        logging.info("build test data:")
        self.test_data = self.build_test_data(self.test_game_path)
        logging.info("read app info:")
        self.dic_app_info = self.read_app_info(self.app_info_path)

        self.process()
        dgl.save_graphs(self.graph_path, self.graph)



    def read_user_id_mapping(self, path):
        mapping = {}
        path_user_id_mapping = "CPGRec/data_exist/user_id_mapping.pkl"
        if os.path.exists(path_user_id_mapping):
            with open(path_user_id_mapping, 'rb') as f:
                mapping = pickle.load(f)

        else:
            count = int(0)
            with open(path,'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line not in mapping.keys():
                        mapping[line] = int(count)
                        count += 1
            with open(path_user_id_mapping, 'wb') as f:
                pickle.dump(mapping, f)    
        return mapping



    def read_app_id_mapping(self, path):
        mapping = {}
        path_app_id_mapping = "/CPGRec/data_exist/app_id_mapping.pkl"
        if os.path.exists(path_app_id_mapping):
            with open(path_app_id_mapping, 'rb') as f:
                mapping = pickle.load(f)

        else:
            count = int(0)
            with open(path,'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if line not in mapping.keys():
                        mapping[line] = int(count)
                        count += 1
            with open(path_app_id_mapping, 'wb') as f:
                pickle.dump(mapping, f)    
        return mapping


    def build_valid_data(self, path):
        intr = {}
        path_valid_data = "/CPGRec/data_exist/valid_data.pkl"
        if os.path.exists(path_valid_data):
            with open(path_valid_data, 'rb') as f:
                intr = pickle.load(f)
        else:
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    line = line.strip().split(',')
                    user = self.user_id_mapping[line[0]]
                    if user not in intr:
                        intr[user] = [self.app_id_mapping[game] for game in line[1:]]
            with open(path_valid_data, 'wb') as f:
                pickle.dump(intr, f)       
        return intr
    


    def build_test_data(self, path):
        intr = {}
        path_valid_data = "/CPGRec/data_exist/test_data.pkl"
        if os.path.exists(path_valid_data):
            with open(path_valid_data, 'rb') as f:
                intr = pickle.load(f)
        else:
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in tqdm(lines):
                    line = line.strip().split(',')
                    user = self.user_id_mapping[line[0]]
                    if user not in intr:
                        intr[user] = [self.app_id_mapping[game] for game in line[1:]]
            with open(path_valid_data, 'wb') as f:
                pickle.dump(intr, f)       
        return intr
    


    def read_game_genre_mapping(self, path):  
        mapping = {}
        path_game_type_mapping = "/CPGRec/data_exist/game_genre_mapping.pkl"
        if os.path.exists(path_game_type_mapping):
            with open(path_game_type_mapping, 'rb') as f:
                mapping = pickle.load(f)

            return mapping

        else:
            mapping_value2id = {}
            count = 0

            with open(path, 'r') as f:       
                lines = f.readlines()    
                for line in tqdm(lines):
                    line = line.strip().split(',')

                    if len(line)>=2 and line[1]!= '' and line[1] not in mapping_value2id:
                        mapping_value2id[line[1]] = count
                        count += 1

                for line in tqdm(lines):
                    line = line.strip().split(',')
                    if self.app_id_mapping[line[0]] not in mapping.keys() and line[1] != '':
                        mapping[self.app_id_mapping[line[0]]] = [line[1]]
                    elif self.app_id_mapping[line[0]] in mapping.keys() and line[1] != '':
                        mapping[self.app_id_mapping[line[0]]].append(line[1])

                    
                for key in tqdm(mapping):
                    mapping[key] = [mapping_value2id[x] for x in mapping[key]] 
                
                mapping_sort = {}
                for key in range(2675):
                    if key not in mapping.keys():
                        mapping_sort[key] = []
                    else:
                        mapping_sort[key] = mapping[key]

                with open(path_game_type_mapping, 'wb') as f:
                    pickle.dump(mapping_sort, f)  

            return mapping



    def read_game_dev_mapping(self, path):  
        mapping = {}
        path_game_type_mapping = "/CPGRec/data_exist/game_dev_mapping.pkl"
        if os.path.exists(path_game_type_mapping):
            with open(path_game_type_mapping, 'rb') as f:
                mapping = pickle.load(f)

            return mapping

        else:
            mapping_value2id = {}
            count = 0

            with open(path, 'r') as f:       
                lines = f.readlines()    
                for line in tqdm(lines):
                    line = line.strip().split(',')

                    if len(line)>=2 and line[1]!= '' and line[1] not in mapping_value2id:
                        mapping_value2id[line[1]] = count
                        count += 1

                for line in tqdm(lines):
                    line = line.strip().split(',')
                    if self.app_id_mapping[line[0]] not in mapping.keys() and line[1] != '':
                        mapping[self.app_id_mapping[line[0]]] = [line[1]]
                    elif self.app_id_mapping[line[0]] in mapping.keys() and line[1] != '':
                        mapping[self.app_id_mapping[line[0]]].append(line[1])

                    
                for key in tqdm(mapping):
                    mapping[key] = [mapping_value2id[x] for x in mapping[key]] 
                
                mapping_sort = {}
                for key in range(len(lines)):
                    if key not in mapping.keys():
                        mapping_sort[key] = []
                    else:
                        mapping_sort[key] = mapping[key]

                with open(path_game_type_mapping, 'wb') as f:
                    pickle.dump(mapping_sort, f)    


            return mapping_sort





    def read_game_pub_mapping(self, path):  
        mapping = {}
        path_game_type_mapping = "/CPGRec/data_exist/game_pub_mapping.pkl"
        if os.path.exists(path_game_type_mapping):
            with open(path_game_type_mapping, 'rb') as f:
                mapping = pickle.load(f)

            return mapping

        else:
            mapping_value2id = {}
            count = 0

            with open(path, 'r') as f:       
                lines = f.readlines()    
                for line in tqdm(lines):
                    line = line.strip().split(',')

                    if len(line)>=2 and line[1]!= '' and line[1] not in mapping_value2id:
                        mapping_value2id[line[1]] = count
                        count += 1

                for line in tqdm(lines):
                    line = line.strip().split(',')
                    if self.app_id_mapping[line[0]] not in mapping.keys() and line[1] != '':
                        mapping[self.app_id_mapping[line[0]]] = [line[1]]
                    elif self.app_id_mapping[line[0]] in mapping.keys() and line[1] != '':
                        mapping[self.app_id_mapping[line[0]]].append(line[1])

                    
                for key in tqdm(mapping):
                    mapping[key] = [mapping_value2id[x] for x in mapping[key]] 
                
                mapping_sort = {}
                for key in range(len(lines)):
                    if key not in mapping.keys():
                        mapping_sort[key] = []
                    else:
                        mapping_sort[key] = mapping[key]

                with open(path_game_type_mapping, 'wb') as f:
                    pickle.dump(mapping_sort, f)    


            return mapping_sort
    



    def read_play_time_rank(self, game_path, time_path):
        path = "/CPGRec/data_exist"
        path_tensor = path+"/tensor_user_game.pth"
        path_dic = path+"/dic_user_game.pkl"

        if os.path.exists(path_tensor) and os.path.exists(path_dic):
            tensor_user_game = torch.load(path_tensor)
            with open(path_dic,"rb") as f:
                dic_user_game = pickle.load(f)
            return tensor_user_game, dic_user_game
        
        else:
            ls = []
            dic_game = {}
            with open(game_path, 'r') as f_game:
                with open(time_path, 'r') as f_time:
                    lines_game = f_game.readlines()
                    lines_time = f_time.readlines()
                    for i in tqdm(range(len(lines_game))):
                        line_game = lines_game[i].strip().split(',')
                        line_time = lines_time[i].strip().split(',')
                        user = self.user_id_mapping[line_game[0]]

                        idx_time_filtered = [i for i in range(len(line_time)) if line_time[i] != r"\N"]
                        line_time_filtered = [float(line_time[i]) for i in idx_time_filtered]

                        if len(line_time_filtered) >0:
                            dic_game[user] = []
                            ar_time = np.array(line_time_filtered)
                            time_mean = np.mean(ar_time)
                        else:
                            continue
                        #

                        for j in range(1, len(line_game)):
                            game = self.app_id_mapping[line_game[j]]
                            dic_game[user].append(game)     
                            time = line_time[j]
                            if time == r'\N':
                                ls.append([user, game, time_mean])     
                            else:
                                ls.append([user, game, float(time)])
                    
                    with open(path_dic, 'wb') as f:
                        pickle.dump(dic_game, f)


            tensor = self.get_time_score(torch.tensor(ls))
            torch.save(tensor, path_tensor)
            return tensor, dic_game


    def get_time_score(self, tensor):
        path = "/CPGRec/data_exist/dic_gametime.pkl"
        if not os.path.exists(path):
            dic = {}
            logging.info("reading time score...")
            for game in tqdm(self.app_id_mapping.values()): 
                mask = (tensor[:,1]==game)
                time = tensor[mask,2]
                dic[game] = time
            self.dic_gametime = dic
            with open(path, 'wb') as f:
                pickle.dump(dic,f)

        
        dic_gametime = get_sigmoid_score(
            path = "/CPGRec/data_exist/dic_gametime.pkl",
            path_score = "/CPGRec/data_exist/dic_gametime_score.pkl"
        )
        logging.info("reading time sigmoid score...")
        for game in tqdm(self.app_id_mapping.values()):
            
            mask = (tensor[:,1]==game)
            tensor[mask,2] = torch.tensor(dic_gametime[game])
        
        return tensor
        



    def game_genre_inter(self, mapping):
        game_type_inter = []
        path_game_genre_inter = "/CPGRec/data_exist/game_genre_inter.pkl"
        if os.path.exists(path_game_genre_inter):
            with open(path_game_genre_inter, 'rb') as f:
                game_type_inter = pickle.load(f)
        else:
            for key in tqdm(list(mapping.keys())):
                for type_key in mapping[key]:
                    game_type_inter.append([key,type_key])
            
            with open(path_game_genre_inter, 'wb') as f:
                pickle.dump(game_type_inter, f)

        return game_type_inter


    def game_dev_inter(self, mapping):
        game_type_inter = []
        path_game_genre_inter = "/CPGRec/data_exist/game_dev_inter.pkl"
        if os.path.exists(path_game_genre_inter):
            with open(path_game_genre_inter, 'rb') as f:
                game_type_inter = pickle.load(f)
        else:
            for key in tqdm(list(mapping.keys())):
                for type_key in mapping[key]:
                    game_type_inter.append([key,type_key])
            
            with open(path_game_genre_inter, 'wb') as f:
                pickle.dump(game_type_inter, f)

        return game_type_inter



    def game_pub_inter(self, mapping):
        game_type_inter = []
        path_game_genre_inter = "/CPGRec/data_exist/game_pub_inter.pkl"
        if os.path.exists(path_game_genre_inter):
            with open(path_game_genre_inter, 'rb') as f:
                game_type_inter = pickle.load(f)
        else:
            for key in tqdm(list(mapping.keys())):
                for type_key in mapping[key]:
                    game_type_inter.append([key,type_key])
            
            with open(path_game_genre_inter, 'wb') as f:
                pickle.dump(game_type_inter, f)

        return game_type_inter




    def read_app_info(self, path):
        path_dic = "CPGRec/data_exist/dic_app_info.pkl"
        if os.path.exists(path_dic):
            with open(path_dic, 'rb') as f:
                dic = pickle.load(f)
            return f
        else:
            df = pd.read_csv(path, header=None)
            games = np.array(list(df.iloc[:,0])).reshape(-1,1)

            prices = np.array(list(df.iloc[:,3]))
            prices /= prices.max()
            prices_mean = prices.mean()
            prices = prices.reshape(-1,1)


            dates = df.iloc[:,4]
            dates = np.array(list(pd.to_datetime(dates).astype(int)))
            dates_mean = dates.mean()
            dates = (dates.astype(float)/dates.max()).reshape(-1,1)
            
            ratings = df.iloc[:,-3].replace(-1,np.nan)
            ratings_mean = ratings.mean()
            ratings = ratings.fillna(ratings_mean).values/100
            ratings = ratings.reshape(-1,1)


            app_info = np.hstack((prices,dates,ratings))
            dic = {}
            for i in range(len(games)):
                dic[self.app_id_mapping[str(games[i][0])]] = app_info[i]

            for game in self.app_id_mapping.keys():
                if game not in games:
                    dic[self.app_id_mapping[game]] = np.array([prices_mean, dates_mean, ratings_mean])
            with open(path_dic,'wb') as f:
                pickle.dump(dic,f)
            return dic





    def process(self):
        logging.info("reading genre,developer,publisher info...")
        self.genre_mapping = self.read_game_genre_mapping(self.genre_path)
        self.genre = self.game_genre_inter(self.genre_mapping)
        self.developer_mapping = self.read_game_dev_mapping(self.developer_path)
        self.developer = self.game_dev_inter(self.developer_mapping)
        self.publisher_mapping = self.read_game_pub_mapping(self.publisher_path)
        self.publisher = self.game_pub_inter(self.publisher_mapping)


        logging.info("reading user item play time...")
        self.user_game, self.dic_user_game = self.read_play_time_rank(self.train_game_path, self.train_time_path)
        
        


        if os.path.exists("/CPGRec/data_exist/graph.bin"):
            graph,_ = dgl.load_graphs("/CPGRec/data_exist/graph.bin")
            graph = graph[0]
            self.graph = graph
            
        else:
            graph_data = {
                ('user', 'friend of', 'user'): (self.friends[:, 0], self.friends[:, 1]),

                ('game', 'developed by', 'developer'): (torch.tensor(self.developer)[:,0], torch.tensor(self.developer)[:,1]),

                ('developer', 'develop', 'game'): (torch.tensor(self.developer)[:,1], torch.tensor(self.developer)[:,0]),

                ('game', 'published by', 'publisher'):(torch.tensor(self.publisher)[:,0], torch.tensor(self.publisher)[:,1]),

                ('publisher', 'publish', 'game'): (torch.tensor(self.publisher)[:,1], torch.tensor(self.publisher)[:,0]),

                ('game', 'genre', 'type'): (torch.tensor(self.genre)[:,0], torch.tensor(self.genre)[:,1]),

                ('type', 'genred', 'game'): (torch.tensor(self.genre)[:,1], torch.tensor(self.genre)[:,0]),

                ('user', 'play', 'game'): (self.user_game[:, 0].long(), self.user_game[:, 1].long()),

                ('game', 'played by', 'user'): (self.user_game[:, 1].long(), self.user_game[:, 0].long())
            }
            graph = dgl.heterograph(graph_data)
            graph.edges['play'].data['time'] = self.user_game[:, 2].to(torch.float32) 
            graph.edges['played by'].data['time'] = self.user_game[:, 2].to(torch.float32)
            

            self.graph = graph
            dgl.save_graphs("/CPGRec/data_exist/graph.bin",[graph])




    def __getitem__(self, i):
        pass

    def __len__(self):
        pass