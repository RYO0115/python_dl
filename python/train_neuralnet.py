import numpy as np
import sys,os
sys.path.append("./dataset/")
from mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize = True, one_hot_label = True)



# ハイパーパラメータ
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

# 学習結果
train_loss_list = []
train_acc_list  = []
test_acc_list   = []

# 1epochあたりの繰り返し数
iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet( input_size = 784, hidden_size = 50, output_size=10 )

for i in range(iters_num):

    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    grad = network.NumericalGradient( x_batch, t_batch) 

    # パラメータの更新
    #
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    # 学習経過の記録
    #
    loss = network.Loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1 epoch ごとに認識精度を計算
    if i % iter_per_epoch == 0 :
        train_acc = network.Accuracy(x_train, t_train)
        test_acc = network.Accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("Traing Acc, Test Acc | ", train_acc, ",", test_acc)

