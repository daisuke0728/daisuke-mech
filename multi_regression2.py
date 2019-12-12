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

    #データの読み込み
    column_names = ['mpg','cylinders','displacement','horsepower','weight',
                    'acceleration', 'model_year', 'origin','name'] 
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
                     names=column_names,delim_whitespace=True)
    
    print(df)
    #######################################################################################################
    #######################################################################################################
    #データ前処理
    
    #欠損値のある行を消去
    #値が?の行の処理:行を削除
    df = df[df['horsepower']!="?"]
    
    #name列を削除
    df = df.drop('name',axis=1)
    
    #originの列をワンホット表現
    origin = df.pop('origin')
    df['USA'] = (origin == 1)*1.0
    df['Europe'] = (origin == 2)*1.0
    df['Japan'] = (origin == 3)*1.0

    #cylinders = 3,5を6に変換
    df.loc[df['cylinders']==3,'cylinders'] = 6
    df.loc[df['cylinders']==5,'cylinders'] = 6

    #displacement列を逆数の値にする
    df.loc[:,'displacement'] = 1/df.loc[:,'displacement']
    
    df = df.astype(float)
    
    #分析用のデータに変換 
    normalize_list = ['cylinders','displacement','horsepower','weight',
                     'acceleration', 'model_year']
    
    analyze_df = df - df.mean()

    #定数項を追加
    analyze_df['const'] = 1.0

    analyze_df.loc[:,normalize_list] = analyze_df.loc[:,normalize_list]/analyze_df.loc[:,normalize_list].std(ddof=0)

    print(df['cylinders'].value_counts())

    """
    #######################################
    #データ可視化実験用エリア

    df.plot(x='USA',y='mpg',kind='scatter')
    plt.show()
    ########################################
    """
    
    #score =0.90391
    #using_columns = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model_year','USA','Europe','const']
    #score = 0.90562
    #using_columns = ['mpg','displacement','horsepower','weight','acceleration','model_year','USA','const']

    #cylinders 変換 displacement 逆数変換後: score = 0.92015
    using_columns = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model_year',
                     'USA','const']

    
    para_len = len(using_columns)

    analyze_df = analyze_df.loc[:,using_columns]
    print(analyze_df.head(5))
    
    #ndarray型に変換
    data = analyze_df.values
    data = data.astype(np.float32)

    #trainデータとtestデータに分割
    train_rate = 0.7
    train_id = np.sort(np.random.choice(data.shape[0],np.int(data.shape[0]*train_rate),replace=False))
    
    #全要素で分析
    train_x = data[train_id,1:]
    train_y = data[train_id,0]
    
    if(train_rate!=1):
        test_x = np.delete(data[:,1:],train_id,axis=0)
        test_y = np.delete(data[:,0],train_id,axis=0)
    else:
        test_x = train_x
        test_y = train_y
        
        
    #重回帰分析
    w = np.array([multi_reg(train_x,train_y)])
    predict_y = np.dot(w,test_x.T)
    score = accuracy(predict_y,test_y)[1,0]

    ##########################################################################
    ##########################################################################
    
    #係数を数値で表示
    print("w: "+str(w))
    print("score: "+str(score))
