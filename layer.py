import numpy as np

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

        #CNN対応用
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        y =  np.dot(x, self.W) + self.b 
        return y
    
    def backward(self, delta):
        # 誤差計算
        dout =  np.dot(delta, self.W.T)        
        # 勾配計算
        self.dW =  np.dot(self.x.T, delta)
        self.db = np.dot(np.ones(len(self.x)), delta) 

        #入力データのサイズに戻す
        dout = dout.reshape(*self.original_x_shape)
        return dout

#Convolution層用の２次元配列を返す関数
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):    
    #input_data : batch,channel_num, y,x
    N, C, H, W = input_data.shape

    #出力サイズを計算
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    #padding 0で補完
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, C*filter_h*filter_w)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

class Convolution:
    def __init__(self, in_dim,out_dim,FH,FW,stride=1, pad=0):
        self.W = 0.01*np.random.randn(out_dim,in_dim,FH,FW)
        self.b = np.zeros(out_dim)
        self.stride = stride
        self.pad = pad

        #初期化
        self.x = None
        self.col = None
        self.col_W = None
        self.param = True
        self.dW = None
        self.db = None
        self.name = 'Convolution'

    def forward(self, x, train_config=True):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        #col_size は(out_h*out_w*N,C*FH*FW) 
        col = im2col(x, FH, FW, self.stride, self.pad)
        #col_W size:(C*FH*FW,FN)
        col_W = self.W.reshape(FN, -1).T
        #out size:(out_h*out_w*N,FN)
        #b size: (1,FN)
        out = np.dot(col, col_W) + self.b
        #out size:(N,FN,out_h,out_w)
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        return dx

    
class MaxPooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

        self.param = False
        self.name = 'MaxPooling'
        
    def forward(self, x, train_config=True):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
    
#過学習を防ぐ:Dropout
class Dropout:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
        self.param = False

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
    def __init__(self,layers,init_weight="std"):
        self.layers = layers
        self.t = None
        
    def forward(self, x, t, train_config=True):   
        #順伝播
        self.t = t
        self.y = x
        for layer in self.layers:
            self.y = layer.forward(self.y, train_config)
        
        #損失関数の計算
        self.loss =  np.sum(-t*np.log(self.y + 1e-10)) / len(x)
        return self.loss
    
    def backward(self):
        #誤差逆伝播 出力層
        dout = (self.y - self.t) / len(self.layers[-1].x)
        
        #中間層を後ろから計算
        for layer in self.layers[-2::-1]:
            #中間層の誤差・勾配計算
            dout =  layer.backward(dout)
        
