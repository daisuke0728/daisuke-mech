#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from argparse import ArgumentParser

#重み行列Wを計算
def multi_reg(x,t):
    W = np.dot(np.dot(np.linalg.inv(np.dot(x.T,x)),x.T),t)
    return W

#scoreとして相関係数を計算
def accuracy(predict,correct):
    return np.corrcoef(predict,correct)

if __name__ == '__main__':

    #乱数固定:学習の結果保存用
    np.random.seed(1)
    
    #コマンドラインからの引数を処理
    argparser = ArgumentParser()
    argparser.add_argument('-norm','--norm_flag',type=int,default=1)
    argparser.add_argument('-num','--refrain_num',type=int,default=1)
    argparser.add_argument('-rate','--train_rate',type=float,default=0.7)
    argparser.add_argument('-graph1','--graph1_flag',type=int,default=0)
    argparser.add_argument('-graph2,','--graph2_flag',type=int,default=1)
    argparser.add_argument('-result','--result_flag',type=int,default=1)

    args = argparser.parse_args()
    print()
    print('コマンドラインからの引数:')
    print(args)
    print()

    #データの読み込み
    column_names = ['mpg','cylinders','displacement','horsepower','weight',
                    'acceleration', 'model_year', 'origin','name'] 
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
                     names=column_names,delim_whitespace=True)

    #欠損値のある行を消去
    #値が?の行の処理:行を削除
    df = df[df['horsepower']!="?"]
    
    #name列を削除
    df = df.drop('name',axis=1)
    
    #originの列をワンホット表現
    origin = df.pop('origin')
    df['USA'] = (origin == 1)*1.
    df['Europe'] = (origin == 2)*1.
    df['Japan'] = (origin == 3)*1.

    #ランクを維持するために'Japan'の列を削除
    #'USA' 'Europe' 'Japan' 'const'の列で縮退が起こる
    df = df.drop('Japan',axis=1)
    
    #ndarray型に変換
    data = df.values
    data = data.astype(np.float32)

    #定数項の列を追加
    data = np.append(data,np.ones(np.shape(data)[0]).reshape(np.shape(data)[0],1),axis=1)
    
    if(args.norm_flag):
        #正規化　originは正規化しない
        data_mean = data[:,:9].mean(axis=0,keepdims=True)
        data_std = np.std(data[:,:7],axis=0,keepdims=True)
        data[:,:9]= (data[:,:9]-data_mean)
        data[:,:7] = data[:,:7]/data_std
    
    #乱数による結果の乱れを平均化
    #N:繰り返し回数
    N = args.refrain_num
    w1 = np.empty((0,9),np.float32)
    score1 = np.array([])
    w2 = np.empty((0,3),np.float32)
    score2 = np.array([])
    
    for i in range(N):
        #trainデータとtestデータに分割
        train_rate = args.train_rate
        train_id = np.sort(np.random.choice(data.shape[0],np.int(data.shape[0]*train_rate),replace=False))
        
        #全要素で分析
        train_x = data[train_id,1:]
        train_y = data[train_id,0]
        #weightとhorsepowerで分析
        X_train = data[np.ix_(train_id,[3,4,9])]
        if(train_rate!=1):
            test_x = np.delete(data[:,1:],train_id,axis=0)
            test_y = np.delete(data[:,0],train_id,axis=0)
            X_test = np.delete(data[:,[3,4,9]],train_id,axis=0)
        else:
            test_x = train_x
            test_y = train_y
            X_test = X_train


        #重回帰分析
        tmp1 = np.array([multi_reg(train_x,train_y)])
        w1 = np.vstack((w1,tmp1))
        predict_y1 = np.dot(tmp1,test_x.T)
        score1 = np.append(score1,accuracy(predict_y1,test_y)[1,0])
        
        tmp2 = np.array([multi_reg(X_train,train_y)])
        w2 = np.vstack((w2,tmp2))
        predict_y2 = np.dot(tmp2,X_test.T)
        score2 = np.append(score2,accuracy(predict_y2,test_y)[1,0])

    w1_mean = np.mean(w1,axis=0)
    w1_std = np.std(w1,axis=0)
    score1_mean = np.mean(score1)
    w2_mean = np.mean(w2,axis=0)
    w2_std = np.std(w2,axis=0)
    score2_mean = np.mean(score2)

    ##########################################################################
    ##########################################################################
    
    #係数を数値で表示
    if(args.result_flag):
        print('w1:全てのパラメータを使用')
        print('w2:horsepowerとweightを使用')
        print("w1_mean: "+str(w1_mean))
        print("w2_mean: "+str(w2_mean))
        print("score1: "+str(score1_mean))
        print("score2: "+str(score2_mean))

    xlabel_list1 = ['cylinders','displacement','horsepower','weight',
                    'acceleration', 'model_year','USA','Europe']
    xlabel_list2 = ['horsepower','weight']
    
    ##########################################################################
    ##########################################################################
    #棒グラフによるWの結果表示
    if(args.graph1_flag):
        #結果の表示
        fig = plt.figure(figsize=(9,6))
        plt.subplots_adjust(wspace=0.2, hspace=1.1)
        
        #9個のパラメータで重回帰分析をしたときの重み行列
        ax1=fig.add_subplot(2,2,1)
        ax1.bar(np.array(range(8)),w1_mean[:-1],tick_label=xlabel_list1,yerr=w1_std[:-1],ecolor="red")
        plt.xticks(rotation=60,size='small')
        ax1.set_title("weight of parameter")
        ax1.set_ylabel("weight")
        #ax1.set_ylim(-10.0,10.0)
        ax1.grid(True)
        
        #2個のパラメータで重景気分析をしたときの重み行列
        ax2=fig.add_subplot(2,2,2)
        ax2.bar(np.array(range(2)),w2_mean[:-1],tick_label=xlabel_list2,yerr=w2_std[:-1],ecolor="red")
        ax2.set_title("weight of parameter")
        ax2.set_ylabel("weight")
        #ax2.set_ylim(-7.0,7.0)
        ax2.grid(True)
        plt.xticks(rotation=60,size='small')
        
        #それぞれの分析の正確性(相関係数を表示)
        ax3 = fig.add_subplot(2,2,3)
        ax3.bar(np.array(range(2)),[score1_mean,score2_mean],tick_label=['9-parameter','2-parameter'])
        ax3.set_title("score of regression")
        ax3.set_ylabel("score")
        plt.xticks(rotation=60,size='small')
        ax3.grid(True)
        ax3.set_ylim(0.7,1)
    
        plt.show()

    #3次元プロット
    if(args.graph2_flag):
        NUMS = 1000

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(1,1,1,projection="3d")
        
        t1 = np.ravel(data[:,3])
        t2 = np.ravel(data[:,4])
        z = np.ravel(data[:,0])
        ax.scatter3D(t1,t2,z,color="aqua",s=4)
        ax.set_xlabel("horsepower")
        ax.set_ylabel("weight")
        ax.set_zlabel("mpg")

        x1 = np.linspace(np.min(t1)-5,np.max(t1)+5,NUMS)
        x2 = np.linspace(np.min(t2)-5,np.max(t2)+5,NUMS)

        X1, X2 = np.meshgrid(x1, x2)
        Y = np.dot(w2_mean,np.array([X1,X2,1]))
        ax.plot_surface(X1,X2,Y,color='lime',alpha=0.5)
        
        plt.show()
