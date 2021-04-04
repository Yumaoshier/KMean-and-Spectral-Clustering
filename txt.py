import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import imageio
import SpectralClustering
import Color


frames =[]
G = None
G1 = None
G0 = None
pos1 = []

def read_txt():
    data = pd.read_csv(r"input\simple-map-dungeon.txt", sep = " ", header=None)
    data = np.array(data)
    #print(data)
    return data

def get_pos0(data):
    global G
    global G0
    global G1
    global pos1
    G = nx.grid_2d_graph(*data.shape)
    G0 = G.copy()
    G1 = G.copy()
    pos1 = {}
    pos0 = []
    for value, node in zip(data.ravel(), G.nodes()):          
        if(value != 0):
            pos1[node] = (node[1], -node[0])
    
            G0.remove_node(node)
        else:
            pos0.append([node[1], -node[0]])
            #pos0[node] = (node[1], -node[0])    
            G1.remove_node(node)
    return pos0
    
def cluster_txt_data(data, numCluster, name, isSpectral = False, k_knn = 4, sigma = 0.1):
    global frames
    frames=[]
    pos0 = get_pos0(data)
    data = G0.nodes()
    #data = np.array(pos0)
    
    if not isSpectral:
        SpectralClustering.startKmean(numCluster, data, name, True)        
    else:
        SpectralClustering.startClustering(numCluster, data, name, k_knn, sigma, True)
    imageio.mimsave("output\\" + name+ ".gif", frames, 'GIF', duration = 0.5)
    

def set_data(indexgroups_pos, dataG):
    Gs = []
    pos0s = []
    for group in indexgroups_pos:
        tempG = nx.Graph()
        tempPos = {}
        for i in group:
            tempPos[(dataG[i][0], dataG[i][1])] = (dataG[i][1], -dataG[i][0])
            #tempPos[(-dataG[i][1], dataG[i][0])] = (dataG[i][0], dataG[i][1])
        tempG.add_nodes_from(tempPos)
        #print(tempG.nodes())
        #print(tempPos)
        Gs.append(tempG)
        pos0s.append(tempPos)
    return Gs, pos0s

def paint(indexgroups_pos, dataG, name, num, k, step):
    '''
    G0 = nx.grid_2d_graph(*data.shape)
    G1 = G0.copy()
  
    pos0 = {}
    pos1 = {}
    for value, node in zip(data.ravel(), G0.nodes()):          
        if(value != 0):
            pos1[node] = (node[1], -node[0])
        else:
            pos0[node] = (node[1], -node[0])    
            G1.remove_node(node)
      
    for node in pos1.keys():
        G0.remove_node(node)
    '''
    Gs, pos0s = set_data(indexgroups_pos, dataG)  
            
    global frames
    plt.figure()
    plt.axis('off')
    plt.title("N=%d,k=%d, iteration:%d" %(num, k, step))  
    nx.draw(G1, pos1, node_color='black', width=11, node_size=100, node_shape = 's')
    count = len(pos0s)
    for i in range(count):       
        nx.draw(Gs[i], pos0s[i], node_color=Color.colors[i%count], width=1, node_size=50, node_shape = 'o')
    plt.savefig("output\\" + name + ".jpg")
    frames.append(imageio.imread('output\\'+name+ '.jpg'))
    #plt.show()
    plt.close()

'''
if __name__ == '__main__':
    data = read_txt()
    get_pos0(data)
    #paint(data)
'''