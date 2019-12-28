import numpy as np

#SGD:確率的勾配降下法
class SGD():
    def __init__(self, lr=0.01):
        #lr:learning rate
        self.lr = lr
        self.network = None
    
    def setup(self, network):
        self.network = network
    
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
        self.v = {'W': [], 'b': []}
        for layer in self.network.layers:
            if layer.param:
                self.v['W'].append(np.zeros_like(layer.W))
                self.v['b'].append(np.zeros_like(layer.b))
    def update(self):
        layer_idx = 0
        for layer in self.network.layers:
            if layer.param:
                self.v['W'][layer_idx] = self.momentum*self.v['W'][layer_idx]-(1-self.momentum)*self.lr*layer.dW
                self.v['b'][layer_idx] = self.momentum*self.v['b'][layer_idx]-(1-self.momentum)*self.lr*layer.db
                layer.W += self.v['W'][layer_idx]
                layer.b += self.v['b'][layer_idx]
                layer_idx += 1

class AdaGrad():
    def __init__(self,lr=0.001,p=0.99,eps=1e-8):
        self.lr = lr
        self.p = p
        self.eps = eps
        self.g = None
        self.network = None
    def setup(self,network):
        self.network = network
        self.g = {'W': [], 'b': []}
        for layer in self.network.layers:
            if layer.param:
                self.g['W'].append(np.zeros_like(layer.W))
                self.g['b'].append(np.zeros_like(layer.b))
    def update(self):
        layer_idx = 0
        for layer in self.network.layers:
            if layer.param:
                self.g['W'][layer_idx] += layer.dW * layer.dW
                self.g['b'][layer_idx] += layer.db * layer.db
                layer.W -= self.lr * layer.dW /np.sqrt((self.g['W'][layer_idx] + self.eps))
                layer.b -= self.lr * layer.db /np.sqrt((self.g['b'][layer_idx] + self.eps))
                layer_idx += 1

class RMSprop():
    def __init__(self,lr=0.01,p=0.99,eps=1e-8):
        self.lr = lr
        self.p = p
        self.eps = eps
        self.v = None
        self.network = None
    def setup(self,network):
        self.network = network
        self.v = {'W': [], 'b': []}
        for layer in self.network.layers:
            if layer.param:
                self.v['W'].append(np.zeros_like(layer.W))
                self.v['b'].append(np.zeros_like(layer.b))
    def update(self):
        layer_idx = 0
        for layer in self.network.layers:
            if layer.param:
                self.v['W'][layer_idx] = self.p * self.v['W'][layer_idx] + (1-self.p) * layer.dW * layer.dW
                self.v['b'][layer_idx] = self.p * self.v['b'][layer_idx] + (1-self.p) * layer.db * layer.db
                layer.W -= self.lr * layer.dW / np.sqrt(self.v['W'][layer_idx] + self.eps)
                layer.b -= self.lr * layer.db / np.sqrt(self.v['b'][layer_idx] + self.eps)
                layer_idx += 1

class AdaDelta():
    def __init__(self,p=0.95,eps=1e-6):
        self.p = p
        self.eps = eps
        self.u = None
        self.v = None
        self.network = None
    def setup(self,network):
        self.network = network
        self.u = {'W': [], 'b': []}
        self.v = {'W': [], 'b': []}
        for layer in self.network.layers:
            if layer.param:
                self.u['W'].append(np.zeros_like(layer.W))
                self.u['b'].append(np.zeros_like(layer.b))
                self.v['W'].append(np.zeros_like(layer.W))
                self.v['b'].append(np.zeros_like(layer.b))
    def update(self):
        layer_idx = 0
        for layer in self.network.layers:
            if layer.param:
                self.v['W'][layer_idx] = self.p*self.v['W'][layer_idx] + (1-self.p) * layer.dW * layer.dW
                self.v['b'][layer_idx] = self.p*self.v['b'][layer_idx] + (1-self.p) * layer.db * layer.db
                delta_W = -layer.dW * np.sqrt(self.u['W'][layer_idx]+self.eps) / np.sqrt(self.v['W'][layer_idx]+self.eps)
                delta_b = -layer.db * np.sqrt(self.u['b'][layer_idx]+self.eps) / np.sqrt(self.v['b'][layer_idx]+self.eps)
                self.u['W'][layer_idx] = self.p * self.u['W'][layer_idx] + (1-self.p) * delta_W * delta_W
                self.u['b'][layer_idx] = self.p * self.u['b'][layer_idx] + (1-self.p) * delta_b * delta_b
                layer.W += delta_W 
                layer.b += delta_b
                layer_idx += 1

class RMSpropGraves():
    def __init__(self,lr=0.0001,p=0.95,eps=1e-4):
        self.lr = lr
        self.p = p
        self.eps = eps
        self.m = None
        self.v = None
        self.network = None
    def setup(self,network):
        self.network = network
        self.m = {'W': [], 'b': []}
        self.v = {'W': [], 'b': []}
        for layer in self.network.layers:
            if layer.param:
                self.m['W'].append(np.zeros_like(layer.W))
                self.m['b'].append(np.zeros_like(layer.b))
                self.v['W'].append(np.zeros_like(layer.W))
                self.v['b'].append(np.zeros_like(layer.b))
    def update(self):
        layer_idx = 0
        for layer in self.network.layers:
            if layer.param:
                self.m['W'][layer_idx] = self.p*self.m['W'][layer_idx]+(1-self.p)*layer.dW
                self.m['b'][layer_idx] = self.p*self.m['b'][layer_idx]+(1-self.p)*layer.db
                self.v['W'][layer_idx] = self.p * self.v['W'][layer_idx] + (1-self.p) * layer.dW * layer.dW
                self.v['b'][layer_idx] = self.p * self.v['b'][layer_idx] + (1-self.p) * layer.db * layer.db
                layer.W -= self.lr * layer.dW / np.sqrt(self.v['W'][layer_idx]-self.m['W'][layer_idx]*self.m['W'][layer_idx]+self.eps)
                layer.b -= self.lr * layer.db / np.sqrt(self.v['b'][layer_idx]-self.m['b'][layer_idx]*self.m['b'][layer_idx]+self.eps)
                layer_idx += 1
                
class Adam():
    def __init__(self,lr=0.001,p1=0.9,p2=0.999,eps=1e-8):
        self.lr = lr
        self.p1 = p1
        self.p2 = p2
        self.eps = eps
        self.m = None
        self.v = None
        self.count = None
        self.network = None
    def setup(self,network):
        self.network = network
        self.m = {'W': [], 'b': []}
        self.v = {'W': [], 'b': []}
        self.count = 1
        for layer in self.network.layers:
            if layer.param:
                self.m['W'].append(np.zeros_like(layer.W))
                self.m['b'].append(np.zeros_like(layer.b))
                self.v['W'].append(np.zeros_like(layer.W))
                self.v['b'].append(np.zeros_like(layer.b))
    def update(self):
        layer_idx = 0
        for layer in self.network.layers:
            if layer.param:
                self.m['W'][layer_idx] = self.p1*self.m['W'][layer_idx] + (1-self.p1) * layer.dW
                self.m['b'][layer_idx] = self.p1*self.m['b'][layer_idx] + (1-self.p1) * layer.db
                self.v['W'][layer_idx] = self.p2*self.v['W'][layer_idx] + (1-self.p2) * layer.dW * layer.dW
                self.v['b'][layer_idx] = self.p2*self.v['b'][layer_idx] + (1-self.p2) * layer.db * layer.db
                tmp_mW = self.m['W'][layer_idx] / (1 - pow(self.p1,self.count))
                tmp_mb = self.m['b'][layer_idx] / (1 - pow(self.p1,self.count))
                tmp_vW = self.v['W'][layer_idx] / (1 - pow(self.p2,self.count))
                tmp_vb = self.v['b'][layer_idx] / (1 - pow(self.p2,self.count))
                layer.W -= self.lr * tmp_mW / np.sqrt(tmp_vW+self.eps) 
                layer.b -= self.lr * tmp_mb / np.sqrt(tmp_vb+self.eps) 
                layer_idx += 1
        self.count += 1

class SMORMS3():
    def __init__(self,lr=0.001,eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.m = None
        self.v = None
        self.s = None
        self.z = None
        self.p = None
        self.count = None
        self.network = None
    def setup(self,network):
        self.network = network
        self.m = {'W': [], 'b': []}
        self.v = {'W': [], 'b': []}
        self.s = {'W': [], 'b': []}
        self.z = {'W': [], 'b': []}
        self.p = {'W': [], 'b': []}
        for layer in self.network.layers:
            if layer.param:
                self.m['W'].append(np.zeros_like(layer.W))
                self.m['b'].append(np.zeros_like(layer.b))
                self.v['W'].append(np.zeros_like(layer.W))
                self.v['b'].append(np.zeros_like(layer.b))
                self.s['W'].append(np.zeros_like(layer.W))
                self.s['b'].append(np.zeros_like(layer.b))
                self.z['W'].append(np.zeros_like(layer.W))
                self.z['b'].append(np.zeros_like(layer.b))
                self.p['W'].append(np.zeros_like(layer.W))
                self.p['b'].append(np.zeros_like(layer.b))
    def update(self):
        layer_idx = 0
        for layer in self.network.layers:
            if layer.param:
                self.s['W'][layer_idx] = 1 + (1 - self.z['W'][layer_idx] * self.s['W'][layer_idx])
                self.s['b'][layer_idx] = 1 + (1 - self.z['b'][layer_idx] * self.s['b'][layer_idx])
                self.p['W'][layer_idx] = 1 / (1 + self.s['W'][layer_idx])
                self.p['b'][layer_idx] = 1 / (1 + self.s['b'][layer_idx])
                self.m['W'][layer_idx] = self.p['W'][layer_idx] * self.m['W'][layer_idx] + (1-self.p['W'][layer_idx]) * layer.dW
                self.m['b'][layer_idx] = self.p['b'][layer_idx] * self.m['b'][layer_idx] + (1-self.p['b'][layer_idx]) * layer.db
                self.v['W'][layer_idx] = self.p['W'][layer_idx] * self.v['W'][layer_idx] + (1-self.p['W'][layer_idx]) * layer.dW * layer.dW
                self.v['b'][layer_idx] = self.p['b'][layer_idx] * self.v['b'][layer_idx] + (1-self.p['b'][layer_idx]) * layer.db * layer.db
                self.z['W'][layer_idx] = self.m['W'][layer_idx] * self.m['W'][layer_idx] / (self.v['W'][layer_idx] + self.eps)
                self.z['b'][layer_idx] = self.m['b'][layer_idx] * self.m['b'][layer_idx] / (self.v['b'][layer_idx] + self.eps)
                
                layer.W -= np.minimum(self.lr,self.z['W'][layer_idx]) * layer.dW / np.sqrt(self.v['W'][layer_idx] + self.eps)
                layer.b -= np.minimum(self.lr,self.z['b'][layer_idx]) * layer.db / np.sqrt(self.v['b'][layer_idx] + self.eps)
                 
                layer_idx += 1
