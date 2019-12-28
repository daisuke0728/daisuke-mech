import numpy as np
from keras.datasets import mnist
import pickle

from preprocess import make_random
from preprocess import data_segmentation
from preprocess import rotate


def mnist_load(new_flag,aug_flag,seg_type,aug_num,range_theta):
    if new_flag:
        #データの読み込み データ数*28*28
        (train_x, train_y), (test_x, test_y) = mnist.load_data()
          
        if aug_flag:
            for i in range(aug_num):
                #データオーグメンテーション
                print("now segmentation {}/{}".format(i+1,aug_num))
                train_seg_x = data_segmentation(train_x,range_theta,seg_type=seg_type)
                train_seg_y = train_y
        
                train_x = np.concatenate([train_x,train_seg_x],0)
                train_y = np.concatenate([train_y,train_seg_y],0)

            
        f = open('train_x.binaryfile','wb')
        pickle.dump(train_x,f)
        f.close
        f = open('train_y.binaryfile','wb')
        pickle.dump(train_y,f)
        f.close
        f = open('test_x.binaryfile','wb')
        pickle.dump(test_x,f)
        f.close
        f = open('test_y.binaryfile','wb')
        pickle.dump(test_y,f)
        f.close
            
    else:
        f1 = open('train_x.binaryfile','rb')
        train_x = pickle.load(f1)
        f2 = open('train_y.binaryfile','rb')
        train_y = pickle.load(f2)
        f3 = open('test_x.binaryfile','rb')
        test_x = pickle.load(f3)
        f4 = open('test_y.binaryfile','rb')
        test_y = pickle.load(f4)
        print("データを読み込みました")
        print("trainデータのデータ数は、{}です。".format(train_x.shape[0]))
    return train_x,train_y,test_x,test_y
