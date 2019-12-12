import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from tqdm import tqdm


#########################################################################
#############関数・クラスの定義##########################################
#########################################################################
#ランダムノイズを加える関数
#percentage:ランダムな値にする割合
def make_random(data,percentage):
    threshold = percentage/100
    w,h = np.shape(data)
    mask = np.random.rand(w,h) < threshold
    data += (np.random.rand(w,h)-data)*mask
    return data

#backwardは逆伝播計算

class Sigmoid:
    def __init__(self):
        self.y = None
        self.param = False
        self.name = 'Sigmoid'
        
    def forward(self, x, train_config=True):
        y = 1 / (1 + np.exp(-x))
        self.y = y
        return y
    
    def backward(self, dout):
        return dout * self.y * (1 -  self.y)

     
class ReLU:
    def __init__(self):
        self.x = None
        self.param = False
        self.name = 'ReLU'
        
    def forward(self, x, train_config=True):
        self.x = x
        return x * (x > 0)
    
    def backward(self, dout):
        return dout * (self.x > 0)

class Softmax:
    def __init__(self):
        self.x = None
        self.y = None
        self.param = False
        self.name = 'Softmax'
        
    def forward(self, x, train_config=True):
        self.x = x
        exp_x = np.exp(x - x.max(axis=1, keepdims=True))
        y = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.y = y
        return y

#in_dimは入力層の次元　out_dimは出力層の次元
#init_weightはWの初期値を決める用の変数

class Linear:
    def __init__(self, in_dim, out_dim, init_weight='std'):
        if init_weight == 'std':
            # 重みの初期値をランダムな値に設定
            self.W = 0.01 * np.random.randn(in_dim, out_dim)
        elif init_weight == 'HeNormal':
            #2.0/in_dimの分散によるガウス分布で初期化
            scale =  np.sqrt(2.0 / in_dim)  
            self.W = scale * np.random.randn(in_dim, out_dim)
        # バイアスの初期化
        self.b = np.zeros(out_dim)
        self.delta = None
        self.x = None
        self.dW = None
        self.db = None
        self.param = True
        self.name = 'Linear'

    def forward(self, x, train_config=True):
        # 順伝播計算
        self.x = x
        y =  np.dot(x, self.W) + self.b 
        return y
    
    def backward(self, delta):
        # 誤差計算
        dout =  np.dot(delta, self.W.T)        
        # 勾配計算
        self.dW =  np.dot(self.x.T, delta)
        self.db = np.dot(np.ones(len(self.x)), delta) 
        
        return dout

#SGD:確率的勾配降下法
class SGD():
    def __init__(self, lr=0.01):
        #lr:learning rate
        self.lr = lr
        self.network = None
    
    def setup(self, network):
        self.network = network
        """
        #debug
        for layer in self.network.layers:
            print(layer.name)
            if layer.param:
                print(layer.W)
        """
    def update(self):
        for layer in self.network.layers:
            if layer.param:
                layer.W -=  self.lr * layer.dW
                layer.b -=   self.lr * layer.db

#モーメント項の考慮したSGD
class Momentum_SGD():
    def __init__(self,lr=0.01,momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        self.network = None
    def setup(self,network):
        self.network = network
        self.v = {'W': [], 'b': []} #vは辞書型、変化量を表す行列
        for layer in self.network.layers:
            if layer.param:
                self.v['W'].append(np.zeros_like(layer.W))
                self.v['b'].append(np.zeros_like(layer.b))
    def update(self):
        layer_idx = 0
        for layer in self.network.layers:
            if layer.param:
                self.v['W'][layer_idx] =  self.momentum * self.v['W'][layer_idx] - self.lr * layer.dW
                self.v['b'][layer_idx] = self.momentum * self.v['b'][layer_idx] - self.lr * layer.db
                layer.W += self.v['W'][layer_idx]
                layer.b += self.v['b'][layer_idx]
                layer_idx += 1

class AdaGrad():
    def __init__(self,lr0=0.001):
        self.lr0 = lr0
        self.v = None
        self.network = None
    def setup(self,network):
        self.network = network
        self.v = {'h': []} #vは辞書型、変化量を表す行列
        for layer in self.network.layers:
            if layer.param:
                self.v['h'].append(0)
    def update(self):
        layer_idx = 0
        for layer in self.network.layers:
            if layer.param:
                self.v['h'][layer_idx] += np.sum(layer.dW*layer.dW)
                layer.W -= self.lr0/(math.sqrt(self.v['h'][layer_idx])+1e-8) * layer.dW
                layer_idx += 1
    
#過学習を防ぐ:Dropout
class Dropout:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
        self.param = False
        self.name = 'Dropout'

    def forward(self, x, train_config=True):
        if train_config:
            #ニューロンを削除するマスクを生成
            self.mask =  np.random.rand(*x.shape) > self.dropout_rate
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_rate)  

    def backward(self, dout):
        return  dout * self.mask
        

class MLP():
    def __init__(self,layers):
        self.layers = layers
        self.t = None
        
    def forward(self, x, t, train_config=True):   
        #順伝播
        self.t = t
        self.y = x
        for layer in self.layers:
            self.y = layer.forward(self.y, train_config)
        
        #損失関数の計算
        self.loss =  np.sum(-t*np.log(self.y + 1e-7)) / len(x)
        return self.loss
    
    def backward(self):
        #誤差逆伝播 出力層
        dout = (self.y - self.t) / len(self.layers[-1].x)
        
        #中間層を後ろから計算
        for layer in self.layers[-2::-1]:
            #中間層の誤差・勾配計算
            dout =  layer.backward(dout)

#データオグメンテーション
#x shape: batch*size*size
def data_segmentation(x,theta_range,seg_type=None):
    shape = np.shape(x)
    input_w = shape[2]
    input_h = shape[1]
    num = shape[0]
    
    #出力行列
    y = np.zeros((num,input_w,input_h))
    #データを回転させたものでかさ増し
    if seg_type == 'rotate':
        for i in range(num):
            #thetaをランダムで決定
            theta = np.random.uniform(theta_range[0], theta_range[1])
            #radianに変換
            rad_theta = np.radians(theta)
            y[i,:,:] = rotate(x[i,:,:],rad_theta)
    return y

#画像を回転させたものを返す
def rotate(input_image,theta):
    # theta:radian
    #image size: 28*28
    image_shape = np.shape(input_image)
    #画像の縦横
    src_w = image_shape[1]
    src_h = image_shape[0]
    pad = 6 #データがはみ出さないように設定
    # 画像のパディング（28x28の画像を(28+pad x 28+pad）の画像になるように周辺を白埋め）
    image = np.zeros((src_h + pad, src_w + pad))
    image[int(pad/2): int(pad/2)+src_h, int(pad/2): int(pad/2) + src_w] = input_image
    
    # 平行移動して中心を(0, 0)にする
    par_array1 = np.asarray([[1, 0, -src_w/2], [0, 1, -src_h/2], [0, 0, 1]]).astype(float)
    # 回転
    rot_array = np.asarray([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]]).astype(float)
    # 平行移動して中心を戻す
    par_array2 = np.asarray([[1, 0, src_w/2], [0, 1, src_h/2], [0, 0, 1]]).astype(float)
    # アフィン行列を計算
    affine_array = np.dot(par_array2,np.dot(rot_array, par_array1))
    # 逆行列を求める
    inv_affine_array = np.linalg.inv(affine_array)
    # 出力画像のピクセル位置のarray
    out_array = np.asarray([[out_x, out_y, 1] for out_x in range(src_w) for out_y in range(src_h)]).T
    # 各出力位置に対応した入力画像の位置のarray
    src_array = np.dot(inv_affine_array, out_array)
    
    #Bilinear補完
    # まず，各出力画素に対応する入力画素をx, yごとに (784, )のshapeのarrayにまとめる
    # → src_xのi番目の要素は，出力画像の(int(i/28), i%28)に位置する画素に対応する入力画素のx座標を表す
    src_x = src_array[0, :].T
    src_y = src_array[1, :].T

    #各入力画素の最近傍に位置する画素（座標の値が整数)
    floor_src_x = np.floor(src_x).astype(int)
    floor_src_y = np.floor(src_y).astype(int)
    
    #最近傍画素の周辺画素の座標のリスト
    x0 = np.clip(floor_src_x, 0, src_w - 1)
    x1 = np.clip(floor_src_x + 1, 0, src_w - 1)
    y0 = np.clip(floor_src_y, 0, src_h - 1)
    y1 = np.clip(floor_src_y + 1, 0, src_h - 1)

    # 周辺画素の画素値を取得
    src_a = input_image[y0, x0]
    src_b = input_image[y1, x0]
    src_c = input_image[y0, x1]
    src_d = input_image[y1, x1]

    # 周辺画素との距離を計算する
    dx = src_x - floor_src_x
    dy = src_y - floor_src_y
    
    # 平均する際にかける重みを計算
    weight_a = (1 - dx) * (1 - dy)
    weight_b = (1 - dx) * dy
    weight_c = dx * (1 - dy)
    weight_d = dx * dy
    
    # 重み付き平均の計算
    output = (weight_a * src_a + weight_b * src_b + weight_c * src_c + weight_d * src_d).reshape(image_shape).T
    
    return output    

def learning(model, optimizer,n_epoch=20,batchsize=100):
    #出力用のリスト
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []

    for epoch in tqdm(range(n_epoch)):
        print('epoch {} : ' .format(epoch), end="")

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
        print('Train loss {:.3f}, Train accuracy {:.3f} | '.format(float(loss), accuracy), end="")
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
        print('Test loss {:.3f}, Test accuracy {:.3f}'.format(float(loss), accuracy))
        test_loss_list.append(float(loss))
        test_acc_list.append(accuracy)
        
    return train_loss_list, train_acc_list, test_loss_list, test_acc_list






#########################################################################
################操作可能なパラメーターの一覧#############################
########################################################################

#データオーグメンテーションを行うか選択
aug_flag = False #True /False
seg_type = 'rotate'
aug_num = 1 #int データオーグメンテーションの回数
range_theta = [-5,10] #rotational angle(°) min to max

#画素値をランダムな値にする割合
random_rate = 0 #percentage (0~100)

#optimizerの選択
optimizer_flag = 'AdaGrad' ## 'SGD' or 'momentum_SGD' or 'AdaGrad'
#optimizer parameter
lr = 0.01 #learning rate
#lr0:AdaGradは大きい方が良い
lr0 = 2.0 #learning rate (init) for AdaGrad
momentum = 0.9 #momentum

#層構造を定義
model_list = [Linear(784, 1000, init_weight='HeNormal'),
              ReLU(),
              Dropout(),
              Linear(1000, 500, init_weight='HeNormal'),
              ReLU(),
              Dropout(dropout_rate=0.3),
              Linear(500,100,init_weight='HeNormal'),
              ReLU(),
              Linear(100, 10, init_weight='HeNormal'),
              Softmax()]

#学習のハイパーパラメーター
n_epoch = 20
batchsize = 100



########################################################################
#################学習の実行#############################################
#######################################################################

#データの前処理

#データの読み込み データ数*28*28
(train_x, train_y), (test_x, test_y) = mnist.load_data()

if aug_flag:
    for i in range(aug_num):
        #データオーグメンテーション
        print("now segmentation {}/{}".format(i+1,aug_num))
        train_seg_x = data_segmentation(train_x,range_theta,seg_type=seg_type)
        train_seg_y = train_y
        
        train_x = np.append(train_x,train_seg_x,axis=0)
        train_y = np.append(train_y,train_seg_y)

#各データの1次元化
X_train = train_x.reshape(np.shape(train_x)[0],-1)
X_test = test_x.reshape(np.shape(test_x)[0],-1)
Y_train = np.identity(10)[train_y]
Y_test = np.identity(10)[test_y]

#データの正規化
X_train = X_train/ 255.0
X_test = X_test / 255.0

#ノイズを加える
X_train = make_random(X_train,random_rate)
X_test  = make_random(X_test,random_rate)

#______________________________________________________________________________________
#______________________________________________________________________________________
#______________________________________________________________________________________

#optimizerの選択
#lrを変更可能
if optimizer_flag == 'SGD':
    optimizer = SGD(lr)
elif optimizer_flag == 'momentum_SGD':
    optimizer = Momentum_SGD(lr, momentum)
elif optimizer_flag == 'AdaGrad':
    optimizer = AdaGrad(lr0)
    
model = MLP(model_list)

#optimizerの設定
optimizer.setup(model)

#n_epochとbatchsizeを変更可能
train_loss_list, train_acc_list, test_loss_list, test_acc_list = learning(model, optimizer,n_epoch,batchsize)


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
