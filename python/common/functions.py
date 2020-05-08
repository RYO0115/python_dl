
import numpy as np


def CrossEntropyError( y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]

    log = np.log(y[np.arange(batch_size), t] + 1e-7)

    return(-np.sum(log) / batch_size)

def NumericalGradient( f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        it.iternext()   
        
    return grad

def SoftMax(x):
    c = np.max(x)
    exp_x = np.exp(x-c) # オーバーフロー対策
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return(y)

def Sigmoid(x):
    h = 1 / (1 + np.exp(-x))
    return(h)
