import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from tqdm import tqdm
import argparse

import data

from optimizer import SGD
from optimizer import Momentum_SGD
from optimizer import AdaGrad
from optimizer import RMSprop
from optimizer import AdaDelta
from optimizer import Adam
from optimizer import RMSpropGraves
from optimizer import SMORMS3

from layer import Linear
from layer import Sigmoid
from layer import ReLU
from layer import Softmax
from layer import Dropout
from layer import MLP

from layer import im2col
from layer import col2im
from layer import Convolution
from layer import MaxPooling

from preprocess import make_random
from preprocess import data_segmentation
from preprocess import rotate

parser = argparse.ArgumentParser()
parser.add_argument('--gpu',help='using gpu or not',type=int,default=0)
parser.add_argument('--download',help='downloading from mnist or use file data',type=int,default=1)
args = parser.parse_args()

gpu = False #gpuを使うかどうか
new_flag = 1
if args.gpu:
    import cupy as xp
else:
    import numpy as xp

#########################################################################
################操作可能なパラメーターの一覧#############################
########################################################################

#データオーグメンテーションを行うか選択
aug_flag = False #True /False
seg_type = 'rotate'
aug_num = 1 #int データオーグメンテーションの回数
range_theta = [-5,5] #rotational angle(°) min to max

#画素値をランダムな値にする割合
random_rate = 0 #percentage (0~100)

#optimizerの選択
optimizer_flag = 'SGD'
## 'SGD' 'momentum_SGD' 'AdaGrad' 'RMSprop' 'AdaDelta' 'Adam' 'RMSpropGraves' 'SMORMS3'

##optimizer parameter
SGD_list = [0.01] #learning rate
momentum_list = [0.01,0.9] #learning rate , momentum
AdaGrad_list = [0.001,0.95,1e-8] #lr,p,eps
RMSprop_list = [0.01,0.99,1e-8] #p,eps
Adadelta_list = [0.95,1e-6] #p,eps
Adam_list = [0.001,0.9,0.999,1e-8] #lr,p1,p2,eps
RMSpropGraves_list = [0.0001,0.95,1e-4] #lr,p,eps
SMORMS3_list = [0.001,1e-8] #lr,eps

#層構造を定義
model_list = [Convolution(1,3,3,3,stride=1,pad=1),
              ReLU(),
              MaxPooling(2,2,stride=1,pad=0),
              Convolution(3,10,3,3,stride=1,pad=1),
              ReLU(),
              #MaxPooling(2,2,stride=2,pad=0),
              #Convolution(10,20,3,3,stride=1,pad=0),
              Linear(7290, 100, init_weight='HeNormal'),
              ReLU(),
              Linear(100,10,init_weight='HeNormal'),
              Softmax()]

#学習のハイパーパラメーター
n_epoch = 20
batchsize = 100*2**(int(aug_flag)*aug_num)


#########################################################################
#############関数・クラスの定義##########################################
#########################################################################

def learning(model, optimizer,n_epoch=20,batchsize=100):
    #出力用のリスト
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    for epoch in tqdm(range(n_epoch)):
        print('epoch {} : ' .format(epoch+1), end="")

        # 訓練
        sum_loss = 0
        pred_y = []
        perm = np.random.permutation(X_train.shape[0])  # 訓練データをランダムにシャッフル

        for i in range(0, X_train.shape[0], batchsize):
            x = X_train[perm[i: i+batchsize]]
            t = Y_train[perm[i: i+batchsize]]

            loss = model.forward(x, t)
            model.backward()
            optimizer.update()

            sum_loss += loss * len(x)
            
            pred_y.extend(np.argmax(model.y, axis=1).tolist())

        loss = sum_loss / X_train.shape[0]

        # accuracy : 予測結果を1-hot表現に変換し，正解との要素積の和を取ることで，正解数を計算できる．
        accuracy = np.sum(np.eye(10)[pred_y] * Y_train[perm]) / X_train.shape[0]
        if gpu:
            accuracy = np.asnumpy(accuracy)
        print('Train loss {:.3f}, Train accuracy {:.4f} | '.format(float(loss), accuracy), end="")
        train_loss_list.append(float(loss))
        train_acc_list.append(accuracy)

        # test
        sum_loss = 0

        pred_y = []
        for i in range(0, X_test.shape[0], batchsize):
            x = X_test[i: i+batchsize]
            t = Y_test[i: i+batchsize]

            sum_loss += model.forward(x, t, train_config=False) * len(x)
            pred_y.extend(np.argmax(model.y, axis=1).tolist())
        loss = sum_loss / X_test.shape[0]

        accuracy = np.sum(np.eye(10)[pred_y] * Y_test) / X_test.shape[0]
        if gpu:
            accuracy = np.asnumpy(accuracy)
        print('Test loss {:.3f}, Test accuracy {:.4f}'.format(float(loss), accuracy))
        test_loss_list.append(float(loss))
        test_acc_list.append(accuracy)
        
    return train_loss_list, train_acc_list, test_loss_list, test_acc_list


########################################################################
#################学習の実行#############################################
#######################################################################

#データの前処理

#データの読み込み size:(データ数*28*28)
train_x,train_y,test_x,test_y = data.mnist_load(new_flag,aug_flag,seg_type,aug_num,range_theta)

if gpu:
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    
#各データの4次元化
X_train = np.zeros((1,train_x.shape[0],train_x.shape[1],train_x.shape[2]))
X_train[0,:,:,:] = train_x
X_train = X_train.transpose(1,0,2,3)
X_test = np.zeros((1,test_x.shape[0],test_x.shape[1],test_x.shape[2]))
X_test[0,:,:,:] = test_x
X_test = X_test.transpose(1,0,2,3)
Y_train = np.identity(10)[train_y]
Y_test = np.identity(10)[test_y]

#データの正規化
X_train = X_train/ 255.0
X_test = X_test / 255.0

#ノイズを加える
#X_train = make_random(X_train,random_rate)
#X_test  = make_random(X_test,random_rate)

#______________________________________________________________________________________
#______________________________________________________________________________________
#______________________________________________________________________________________

#optimizerの選択
#lrを変更可能
if optimizer_flag == 'SGD':
    optimizer = SGD(lr=SGD_list[0])
elif optimizer_flag == 'momentum_SGD':
    optimizer = Momentum_SGD(lr=momentum_list[0], momentum=momentum_list[1])
elif optimizer_flag == 'AdaGrad':
    optimizer = AdaGrad(lr=AdaGrad_list[0],p=AdaGrad_list[1],eps=AdaGrad_list[2])
elif optimizer_flag == 'RMSprop':
    optimizer = RMSprop(lr=RMSprop_list[0],p=RMSprop_list[1],eps=RMSprop_list[2])
elif optimizer_flag == 'AdaDelta':
    optimizer = AdaDelta(p=Adadelta_list[0],eps=Adadelta_list[1])
elif optimizer_flag == 'Adam':
    optimizer = Adam(lr=Adam_list[0],p1=Adam_list[1],p2=Adam_list[2],eps=Adam_list[3])
elif optimizer_flag == 'RMSpropGraves':
    optimizer = RMSpropGraves(lr=RMSpropGraves_list[0],p=RMSpropGraves_list[1],eps=RMSpropGraves_list[2])
elif optimizer_flag == 'SMORMS3':
    optimizer = SMORMS3(lr=SMORMS3_list[0],eps=SMORSM3_list[1]) 
    
model = MLP(model_list)

#optimizerの設定
optimizer.setup(model)

#n_epochとbatchsizeを変更可能
train_loss_list, train_acc_list, test_loss_list, test_acc_list = learning(model, optimizer,n_epoch,batchsize)

if gpu:
    #cpuへの変換
    train_loss_list = np.asnumpy(train_loss_list)
    train_acc_list = np.asnumpy(train_acc_list)
    test_loss_list = np.asnumpy(test_loss_list)
    test_acc_list = np.asnumpy(test_acc_list)
    import numpy as np # for plot

#______________________________________________________________________________________
#______________________________________________________________________________________
#______________________________________________________________________________________


# 結果のプロット
#lossのプロット
plt.subplot(1,2,1)
plt.plot(np.arange(len(train_loss_list)), np.asarray(train_loss_list), label='train')
plt.plot(np.arange(len(test_loss_list)), np.asarray(test_loss_list), label='test')
plt.title('loss function')
plt.legend()

#accのプロット
plt.subplot(1,2,2)
plt.plot(np.arange(len(train_acc_list)), np.asarray(train_acc_list), label='train')
plt.plot(np.arange(len(test_acc_list)), np.asarray(test_acc_list), label='test')
plt.title('acc function')
plt.show()
