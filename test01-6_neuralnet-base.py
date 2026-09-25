from common import TwoLayerNet
import numpy as np


print("create model:")
net = TwoLayerNet(input_size=784, hidden_size=10, output_size=10)
print(net.params['W1'].shape) # (784, 10)
print(net.params['b1'].shape) # (10,)
print(net.params['W2'].shape) # (10, 10)
print(net.params['b2'].shape) # (10,)

print("predict:")
x = np.random.rand(100, 784) # 伪输入数据（100笔）
y = net.predict(x)

print("numerical_gradient:")
x = np.random.rand(100, 784) # 伪输入数据（100笔）
t = np.random.rand(100, 10) # 伪正确解标签（100笔）

grads = net.numerical_gradient(x, t) # 计算梯度

print(grads['W1'].shape) # (784, 10)
print(grads['b1'].shape) # (10,)
print(grads['W2'].shape) # (10, 10)
print(grads['b2'].shape) # (10,)




