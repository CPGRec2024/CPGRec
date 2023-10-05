import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_percent', default = 0.8, type = float,
                        help = 'training_percent')
    parser.add_argument('--embed_size', default = 32, type = int,
                        help = 'embedding size for all layer')
    parser.add_argument('--lr', default = 0.03, type = float,
                        help = 'learning rate')
    parser.add_argument('--epoch', default = 1000, type = int,
                        help = 'epoch number')
    parser.add_argument('--early_stop', default = 10, type = int,
                        help = 'early_stop validation')
    parser.add_argument('--batch_size', default = 1024, type = int,
                        help = 'batch size')
    parser.add_argument('--layers', default = 1, type = int,
                        help = 'layer number')
    parser.add_argument('--gpu', default = 0, type = int,
                        help = '-1 for cpu, 0 for gpu:0')
    parser.add_argument('--k', default = [5,10,20], type = list,
                        help = 'negative sampler number for each node')
    parser.add_argument('--K', default = 6.5, type = float,
                        help = 'hyper-parameter for negative score reweighting')
    parser.add_argument('--gamma', default=80.0, type=float,
                        help='hyper-parameter for aggregation weight')
    parser.add_argument('--layers_and', default=2, type=int,
                        help='hyper-parameter for layer number')
    parser.add_argument('--layers_or', default=4, type=int,
                        help='hyper-parameter for layer number')
    parser.add_argument('--layers_user_game', default=2, type=int,
                        help='hyper-parameter for layer number')
    parser.add_argument('--attention_and', default=True, type=bool,
                        help='hyper-parameter for attention of and layers')
    parser.add_argument('--param_decay', default=0.1, type=float,
                        help='hyper-parameter for decay')
    parser.add_argument('--edge_node_weight', default=True, type=bool,
                        help='hyper-parameter for diversification in bipartite')
    args = parser.parse_args()
    return args
