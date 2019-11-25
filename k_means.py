#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

################################################################
#グローバル変数

plt_flag = False
#重心の初期設定の仕方の選択(1~3)
initial_flag = 3

################################################################

#乱数固定(結果の保持用)
np.random.seed(80)

################################################################

#k-means x:data n:class_num
#dataとして、データ数*column数を想定

################################################################

#k-meansの重心座標　初期値設定の関数
#kkz法の実装
def KKZ(x,n):
    center = np.zeros(x.shape[1]*n).reshape(n,x.shape[1])
    #１つ目はランダムに選択
    init_id = np.random.choice(x.shape[0],1)
    #print(init_id)
    id_list = np.array(range(x.shape[0]))
    #print([id_list==init_id])
    id_list = id_list[~(id_list==init_id)]
    #print(id_list.shape[0])
    center[0,:] = x[init_id,:]
    counter = 1
    
    while(n>counter):
        #今まで存在するcecnterの点から最も遠い点を新しい重心座標とする
        memory = np.zeros(2*id_list.shape[0]).reshape(2,id_list.shape[0])
        for i in range(id_list.shape[0]):
            tmp = np.min(np.linalg.norm(x[i]-center[:counter],axis=1))
            memory[0,i] = tmp
            memory[1,i] = id_list[i]
        new_id  = memory[1,np.argmax(memory,axis=1)[0]]
        id_list = id_list[~(id_list==new_id)]
        #print("counter: {} new_id is {}".format(counter,new_id))
        center[counter]=x[np.int(new_id)]
        counter+=1
    #print(center)
    #print("n is {}".format(n))
    return center

#k-meansの重心座標　初期値設定の関数
#k-means++法
def k_means_plus(x,n):
    center = np.zeros(x.shape[1]*n).reshape(n,x.shape[1])
    #１つ目はランダムに選択
    init_id = np.random.choice(x.shape[0],1)
    id_list = np.array(range(x.shape[0]))
    id_list = id_list[~(id_list==init_id)]
    center[0,:] = x[init_id,:]
    counter = 1

    while(n>counter):
        #今まで存在するcecnterの点からの最小距離を確率として新しい重心座標を選ぶ
        memory = np.zeros(2*id_list.shape[0]).reshape(2,id_list.shape[0])
        for i in range(id_list.shape[0]):
            tmp = np.min(np.linalg.norm(x[i]-center[:counter],axis=1))
            memory[0,i] = tmp
            memory[1,i] = id_list[i]
        memory_sum = np.sum(memory[0,:])
        memory[0,:] = memory[0,:]/memory_sum
        new_id = np.random.choice(id_list,p=memory[0,:])
        center[counter]=x[np.int(new_id)]
        counter+=1
    #print(center)    
    return center

###################################################################################
##############################k-meansの実装########################################
###################################################################################

def k_means(x,n):

    global initial_flag
    
    if initial_flag == 1:
        #データの中からランダムに選ぶ
        num = np.random.choice(x.shape[0],n)
        center=x[num,:]
    elif initial_flag == 2:
        #KKZ法
        center = KKZ(x,n)
    elif initial_flag == 3:
        center = k_means_plus(x,n)
        
    else:
        #０中心の乱数
        center= np.random.randn(n*x.shape[1]).reshape(n,x.shape[1])
    
    #ans center_oldを初期化
    center_old = np.zeros(n*x.shape[1]).reshape(n,x.shape[1])
    
    #重心が移動しなくなるまで繰り返す
    while(np.all(center_old != center)):
        center_old = center
        #各データについて一番近い重心のクラスを取得
        ans = np.array([np.argmin(np.linalg.norm(x[i]-center,axis=1)) for i in range(x.shape[0])])
        #新しい重心座標を決定
        center = np.array([np.mean(x[np.where(ans==i)],axis=0) if len(ans[ans==i])!=0 else center_old[i] for i in range (n)])
    return ans, center

########################################################################################
########################################################################################

##結果表示
def plot_by_column(data,center,ans_label,n):

    #すべてのcolumnの組み合わせに対してグラフを表示
    columns = ["sepal length", "sepal width", "petal length", "petal width", "class_label"]
    colors = {0:"blue",1:"yellow",2:"green",3:"black",4:"orange",5:"purple"}

    ref = pd.DataFrame(data,columns=columns[:4])
    ref['class_label']=ans_label
    
    plt.figure(figsize=(9,9))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)

    for i in range(4):
        for j in range(4):
            axis1= i
            axis2= j
        
            plt.subplot(4,4,i*4+j+1)
            if (i != j):
                plt.title("k-means")
                plt.xlabel(columns[axis1])
                plt.ylabel(columns[axis2])
                plt.grid(True)
                for f in ref['class_label'].unique():
                    plt.scatter(ref.loc[ref.class_label==f,columns[axis1]],ref.loc[ref.class_label==f,columns[axis2]],
                                c=colors[f],label=f,s=3)
                plt.scatter(center[:,axis1],center[:,axis2],s=10,c="r",marker="D")
    plt.show()    
    return 0

#提出用のプロット
def plot_submit(data,center,ans_label,n):

    global plt_flag
    #グラフを描画する列を決定
    axis1 = 1
    axis2 = 2
    
    columns = ["sepal length", "sepal width", "petal length", "petal width", "class_label"]
    colors = {0:"blue",1:"yellow",2:"green",3:"black",4:"orange",5:"purple"}

    ref = pd.DataFrame(data,columns=columns[:4])
    ref['class_label']=ans_label

    if plt_flag != True:
        plt.figure(figsize=(9,9))
        plt.subplots_adjust(wspace=0.4, hspace=0.3)
        plt_flag = True

    plt.subplot(2,3,n-1)
    plt.title("class_num = %i" %n)
    plt.xlabel(columns[axis1])
    plt.ylabel(columns[axis2])
    plt.grid(True)
    for f in ref['class_label'].unique():
        plt.scatter(ref.loc[ref.class_label==f,columns[axis1]],ref.loc[ref.class_label==f,columns[axis2]],
                    c=colors[f],label=f,s=3)
    plt.scatter(center[:,axis1],center[:,axis2],s=10,c="r",marker="D",label="center")

    if n==6:
        #判例を出す位置を左上に調整
        plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=1, )
        plt.show()
    return 0

#考察用のグラフ
def plot_for_consideration(data):

    global initial_flag

    #グラフを描画する列を決定
    axis1 = 1
    axis2 = 2
    
    columns = ["sepal length", "sepal width", "petal length", "petal width", "class_label"]
    colors = {0:"blue",1:"yellow",2:"green"}
    colors_class = {'Iris-setosa':"blue",'Iris-versicolor':"yellow",'Iris-virginica':"green"}
    title_list = {1:"random select",2:"KKZ",3:"k-means++"}

    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    
    #元データの分布をプロット
    plt.subplot(2,2,1)
    plt.title("reference data")
    plt.xlabel(columns[axis1])
    plt.ylabel(columns[axis2])
    plt.grid(True)
    
    ref = pd.DataFrame(data,columns=columns[:4])
    ref['class_label']=label
    
    for f in ref['class_label'].unique():
        plt.scatter(ref.loc[ref.class_label==f,columns[axis1]],ref.loc[ref.class_label==f,columns[axis2]],
                    c=colors_class[f],label=f,s=3)
        
    for i in range(3):
        initial_flag = i+1
        plt.subplot(2,2,i+2)
        plt.title(title_list[i+1])
        plt.xlabel(columns[axis1])
        plt.ylabel(columns[axis2])
        plt.grid(True)
        class_label,center = k_means(data,3)
        ref['class_label']=class_label
        for f in ref['class_label'].unique():
            plt.scatter(ref.loc[ref.class_label==f,columns[axis1]],ref.loc[ref.class_label==f,columns[axis2]],
                        c=colors[f],label=f,s=3)
            plt.scatter(center[:,axis1],center[:,axis2],s=10,c="r",marker="D",label="center")
        
    plt.show()
    

##################################################################################
##########################メイン文################################################
##################################################################################

if __name__ == '__main__':
    #データの読み込み
    df = pd.read_csv(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
    df.columns = ["sepal length", "sepal width", "petal length", "petal width", "class"]

    #ndarray型に変換
    data = df.drop("class",axis=1).values
    label = df['class'].values

    #GAUSS分布に変換
    data_mean = data.mean(axis=0,keepdims=True)
    data_std = np.std(data,axis=0,keepdims=True)
    data = (data-data_mean)/data_std

    #クラス数を2~6で実行
    for k in range(2,7):
        class_label,center = k_means(data,k)
        
        #結果のプロット
        #plot_by_column(data,center,class_label,k)
        plot_submit(data,center,class_label,k)

    #考察用のグラフ生成
    plot_for_consideration(data)

"""
class:
Iris-setosa,Iris-versicolor,Iris-virginica
"""
