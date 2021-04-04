#%matplotlib inline

import seaborn as sns
import numpy as np
import pandas as pd
from itertools import chain
import random
import math
import matplotlib.pylab as plt
import imageio
#from sklearn.cluster import KMeans
#from sklearn.cluster import SpectralClustering
from sklearn import datasets
from tkinter import filedialog
import txt
import Color



sns.set_style('darkgrid', {'axes.facecolor':'.9'})
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep') 


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

rs = np.random.seed(25)

step = 0
frame = []
df = []

def genTwoCircles(n_samples=1000):
    X,y = datasets.make_circles(n_samples, factor=0.5, noise=0.05)
    return X, y



def get_Weight_Matrix(data):    
   # print(data)
    S = np.zeros((len(data), len(data)))
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            #if(S[i][j] == 0):
            #np.sqrt(np.sum((data[i] - data[j]) ** 2))
            S[i][j] = distance(data[i], data[j])
            S[j][i] = S[i][j]
    return S

def get_AdjacentKNN_Matrix(S, k=5, sigma = 1, isTxt = False):
   # N = len(S)   
    #print(sigma)
    A = np.zeros(S.shape)
    for i in range(S.shape[0]):
        index_S = np.argsort(S[i])[:k]
        for j in index_S:
            #if(A[i][j] == 0):
            if isTxt:
                A[i][j] = 1
            else:               
                A[i][j] = np.exp(-S[i][j]/2/(sigma * sigma))
            A[j][i] = A[i][j]
    return A

def get_LaplacianNormal_Matrix(A):
    D = np.sum(A, axis = 1)
    L = np.diag(D) - A
    sqrtD = np.diag(1.0 / (D ** 0.5))
    L = np.dot(np.dot(sqrtD, L), sqrtD)
    return L

def distance(x, y):
    return np.sqrt(np.abs(x[0] - y[0]) ** 2 + np.abs(x[1] - y[1]) ** 2)

def distance_advanced(x, y):
    result = 0.0
    #print('x:', x)
    #print('y:', y)
    for i in range(len(x)):      
        result += np.power(np.abs(x[i] - y[i]), 2)
    #result = np.sqrt(result)
    #print('result:', result)
    return result


def Kmeans(eigenFeature, k, data, name, isTxt = False):
   
    index0_Centers = random.sample(range(eigenFeature.shape[0]), k)
    #centers = []
    #n = eigenFeature.shape[0] // k -1
    #for i in range(k):
        #centers.append(eigenFeature[(i + 1) * n])
    centers = np.array([eigenFeature[i] for i in index0_Centers])
    #centers = np.array(centers)
    #centers.sort(axis=0)
    centers = centers[centers[:, 0].argsort()]
 
    global step
    global frame
    pre_error = 0
    error_count = 0
    while True:
        dataIndex_groups = [[] for i in range(k)]
        for i in range(eigenFeature.shape[0]):                      
            cur_Point = eigenFeature[i]          
            #distances = [np.sum(np.abs(np.power((centers[j] - cur_Point), 2))) for j in range(k)]
            distances = [distance_advanced(centers[j], cur_Point) for j in range(k)]
            #print(distances)
            minIndex = distances.index(min(distances))
            #print(minIndex)
            dataIndex_groups[minIndex].append(i)
 
        step +=1
        centers_new = []
        maxIndex = 0
        maxCount = 0
       
        for i in range(k):
            count = len(dataIndex_groups[i])
           
            if count > 0:              
                new_center = np.sum(eigenFeature[dataIndex_groups[i]], axis = 0) / count           
                centers_new.append(new_center)    
                if count > maxCount:
                    maxCount = count
                    maxIndex = i
        
        leave_count = len(centers) - len(centers_new)
        
        if leave_count > 0:
            index0_Centers = random.sample(range(eigenFeature.shape[0]), leave_count)
            for i in index0_Centers:
                centers_new.append(eigenFeature[i])
            #centers_new.append(np.array([eigenFeature[i] for i in index0_Centers]))
            '''
            n = eigenFeature.shape[0] // leave_count - 1
            for i in range(leave_count):             
                centers_new.append(eigenFeature[(i + 1) * n])
            
            maxgroup = eigenFeature[dataIndex_groups[maxIndex]]
            #maxgroup.sort(axis = 0)
            n = (len(maxgroup)//leave_count) - 1
            for i in range(leave_count):             
                centers_new.append(maxgroup[(i+1) * n])
            '''
        #print(centers_new)
        centers_new = np.array(centers_new)    
        centers_new = centers_new[centers_new[:, 0].argsort()]
       
        if not isTxt:          
            plt.figure()
            plt.title("N=%d,k=%d, iteration:%d" %(data.shape[0], k, step))      
            
            for i in range(k):
                x = []
                y = []
                for j in dataIndex_groups[i]:
                    x.append(data[j][0])
                    y.append(data[j][1])          
                plt.scatter(x, y, s=10, c=Color.colors[i%k])          
                #plt.plot(x, y, color[i%4])
           
            plt.savefig('output\\' + name + '.jpg')
            frame.append(imageio.imread('output\\'+name+ '.jpg'))
            plt.close()
        else:
            txt.paint(dataIndex_groups, data, name, data.shape[0], k, step)
        '''
        print("centers:", centers)
        print("centers_new:", centers_new)
        print(centers - centers_new)
        '''
        temperror = np.sum(np.abs(centers - centers_new))
        #print(temperror)
        #print('pre:', pre_error)
        #not math.isclose(temperror, 0, rel_tol=1e-5)
        if(temperror != 0):
            centers = centers_new            
            if(math.isclose(temperror, pre_error)):
                if(error_count < 5):
                    error_count += 1                               
                else:
                    return dataIndex_groups 
            elif step > 50:
                return dataIndex_groups 
            else:
                error_count = 0
                pre_error = temperror
        else:
            return dataIndex_groups
      
def plotRes(data, clusterResult, clusterNum):
    n = len(data)
    for i in range(clusterNum):
        _color = Color.colors[i % len(Color.colors)]
        x1= []; y1=[]
        for j in range(n):
            if clusterResult[j] == i:
                x1.append(data[j,0])
                y1.append(data[j, 1])
        plt.scatter(x1, y1, c=_color, marker='+')
    plt.show()



    
def get_Excel():
    global df
    importpath = filedialog.askopenfilename()
    read_file = importpath.read_excel(importpath)
    #df = pd.read_excel(r'D:\2020-goldsmiths-AI\CW1\K-means\3.xlsx')    
    df = np.array(read_file)
    #print(df)
    #return df
    
    
def startClustering(numCluster, data, name = "spectralClustering", k_knn = 4, sigma = 0.1, isTxt = False):  
    global step
    global frame
    step = 0
    frame =[]
    
    data = np.array(data)
    #data = data[data[:, 0].argsort()]
   
    W = get_Weight_Matrix(data)
    #print(W)
    #sigma = 0.1
    #k_knn = 4
    #print(k_knn, sigma)
    A = get_AdjacentKNN_Matrix(W, k_knn, sigma, isTxt)
    Lnormal = get_LaplacianNormal_Matrix(A)
    eigenValues, eigenVectors = np.linalg.eig(Lnormal)
   # print(eigenValues)
    
    #eigen_dimen = 2 numCluster
    index_Values = []
    if isTxt:
        index_Values = np.argsort(eigenValues)[:k_knn]   
    else:
        index_Values = np.argsort(eigenValues)[:k_knn]   

    #print(index_Values)
    eigenFeature = eigenVectors[:, index_Values]
   
    groups_index = Kmeans(eigenFeature, numCluster, data, name, isTxt)
    if not isTxt:
        imageio.mimsave("output\\" + name+ ".gif", frame, 'GIF', duration = 0.5)
    
    
def startKmean(numCluster, data, name = "KMeansClustering", isTxt = False):
    global step
    global frame
    step = 0
    frame =[]
    
    data = np.array(data)
    
    groups_index = Kmeans(data, numCluster, data, name, isTxt)
    if not isTxt:
        imageio.mimsave('output\\'+ name+ '.gif', frame, 'GIF', duration = 0.5)
    
    

def fake_main():
    
    '''
    x, y = np.loadtxt('2.csv', delimiter=',', unpack=True)
    data = list(zip(x, y))
    data = np.array(data)
    '''
    #data = get_Excel()
    k_count = 2
   
    data, label = genTwoCircles(n_samples=500)
    
    name = "KMeansClustering"
    groups_index = Kmeans(data, k_count, data, name)
    step = 0
    frame =[]
    
    W = get_Weight_Matrix(data)
    #print(W)
    sigma = 0.1
    k_knn = 5
    A = get_AdjacentKNN_Matrix(W, k_knn, sigma)
    Lnormal = get_LaplacianNormal_Matrix(A)
    eigenValues, eigenVectors = np.linalg.eig(Lnormal)
   # print(eigenValues)
    
    #eigen_dimen = 2
    index_Values = np.argsort(eigenValues)[0:k_count]   
    #print(index_Values)
    eigenFeature = eigenVectors[:, index_Values]
    #print(eigenFeature)
    #eigenFeature = np.c_[eigenFeature, np.zeros(eigenFeature.shape[0])]
    name = "spectralClustering"
    groups_index = Kmeans(eigenFeature, k_count, data, name)
    '''
    clf = KMeans(n_clusters=k_count)
    label = clf.fit(data).labels_
    plotRes(data, label, k_count)
    label2 = SpectralClustering(gamma=0.5, n_clusters = k_count, affinity='rbf').fit_predict(data)
    plotRes(data, label2, k_count)
    '''
    imageio.mimsave('spectral clustering.gif', frame, 'GIF', duration = 0.5)
    

    
  
    

