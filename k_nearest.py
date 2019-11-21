import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from argparse import ArgumentParser

#k近傍法でクラスを返す
#x:dataset y:class,
def k_nearest(x,y,k):
    class_list=np.array([])
    #各データについて2乗誤差を取る
    for i in range(x.shape[0]):
        #2乗誤差の小さい順にk個のidを抽出(1番小さいのはx[i]-x[i]なので、それは除く)
        ans_id = np.argsort(np.linalg.norm(x[i,:]-x,axis=1))[1:k+1]
        #normが小さい方からk個の教師データを個数がvalueの辞書型にする
        label_dict = Counter(y[ans_id])
        #valueが最も多いクラスを抽出(複数可)
        max_key_list = [j[0] for j in label_dict.items() if j[1]==max(label_dict.values())]
        #max_key_listから一つのクラスを抽出
        class_list = np.append(class_list,max_key_list[np.random.randint(0,len(max_key_list),1)[0]])
    #全データの予測クラスを返す
    return class_list
    
#正答率を計算　
def accuracy(predict,correct):
    return np.array([predict == correct]).astype(np.int).mean()

if __name__ == '__main__':
    #コマンドラインからの引数を取得
    argparser = ArgumentParser()
    argparser.add_argument('-init','--init_flag',type=int,default=1)
    args = argparser.parse_args()
    
    #データの読み込み
    df = pd.read_csv(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None)
    df.columns = ["sepal length", "sepal width", "petal length", "petal width", "class"]

    #ndarray型に変換
    #説明変数
    data = df.drop("class",axis=1).values
    
    #正解ラベル
    data_y = df['class'].values
    
    #データの正規化方法を指定
    if(args.init_flag==1):
        #GAUSS分布に変換
        data_mean = data.mean(axis=0,keepdims=True)
        data_std = np.std(data,axis=0,keepdims=True)
        data = (data-data_mean)/data_std
        
    elif(args.init_flag==2):
        #minを0にmaxを1になるように変換
        data_min = np.min(data,axis=0,keepdims=True)
        data_max = np.max(data,axis=0,keepdims=True)
        data = (data-data_min)/(data_max-data_min)
    else:
        print("nothing has done for data")

    #k is 1~149
    K = 30
    acc = np.zeros(K)
    
    #k近傍法の実行
    for k in range (1,K+1):       
        predict_label = k_nearest(data,data_y,k)
        acc[k-1] = accuracy(predict_label,data_y)
        #print("accuracy:: k:{} , 正答数:{}/{}".format(k,int(acc[k-1]*np.shape(data)[0]),np.shape(data)[0]))

    #結果の出力
    plt.plot(np.array(range(1,K+1)),acc)
    plt.title("k-nearest neighbor")
    plt.xlabel("k")
    plt.ylabel("accuracy")
    plt.grid(True)
    plt.show()

"""
class:
Iris-setosa,Iris-versicolor,Iris-virginica
"""



