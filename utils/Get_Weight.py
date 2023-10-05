import dgl
import torch




def get_weight(theta1, theta2, theta3):
    
    torch.manual_seed(2023)
    path = "/CPGRec/data_exist/graph.bin"

    graph = dgl.load_graphs(path)[0]
    graph = graph[0]

    graph = dgl.edge_type_subgraph(graph, etypes=['played by'])

    outdeg_game = graph.out_degrees(graph.edges(etype = ('game','played by','user'))[0],etype = 'played by').float()


    quantile_low = 50000    #0.2 quantile of outdeg_game
    quantile_high = 850000  #0.8 quantile of outdeg_game

    idx_low = (outdeg_game<=quantile_low)
    idx_high = (outdeg_game>quantile_high)
    torch.save(idx_high, "/CPGRec/data_exist/idx_high.pth")

    weight_edge = torch.ones_like(outdeg_game)
    weight_edge[idx_high] = weight_edge[idx_high] * theta1


    path_weight_edge = "/CPGRec/data_exist/weight_edge.pth"
    torch.save(weight_edge, path_weight_edge)



    outdeg_game_2 = graph.out_degrees( graph.nodes(ntype = 'game'),etype = 'played by').float()
    weight_node = torch.ones_like(outdeg_game_2)

    quantile_low_2 = 570    #0.2 quantile of outdeg_game_2
    quantile_high_2 = 33000 #0.8 quantile of outdeg_game_2

    path_cold =  "/CPGRec/data_exist/game_cold.pth"
    path_hot =  "/CPGRec/data_exist/game_hot.pth"
    torch.save(torch.where(outdeg_game_2 <= quantile_low_2), path_cold)
    torch.save(torch.where(outdeg_game_2 >= quantile_high_2), path_hot)


    idx_low = (outdeg_game_2<=quantile_low_2)
    idx_high = (outdeg_game_2>quantile_high_2)
    weight_node[idx_low] = weight_node[idx_low] * theta3
    weight_node[idx_high] = weight_node[idx_high] * theta2

    path_weight_node = "/CPGRec/data_exist/weight_node.pth"
    torch.save(weight_node, path_weight_node)




if __name__ == "__main__":
    get_weight(theta1=80, theta2=0.5, theta3=3)