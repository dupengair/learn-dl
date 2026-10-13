from common import SimpleConvNet, numerical_gradient
import numpy as np

network = SimpleConvNet(input_dim=(1, 10, 10),
                        conv_param={'filter_num': 2, 'filter_size': 3,
                                    'pad': 1, 'stride': 1},
                        hidden_size=5, output_size=3,
                        weight_init_std=0.01)

x = np.random.rand(2, 1, 10, 10)
t = np.array([[1,0,0], [0,1,0]])  # one-hot

grad_backprop = network.gradient(x, t)
grad_num = network.numerical_gradient(x, t)

for key in grad_backprop.keys():
    diff = np.mean(np.abs(grad_backprop[key] - grad_num[key]))
    print(key + ":" + str(diff))