import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import time
import tensorflow as tf
print(tf.__version__)    #2.4.1

#Initialization of Tensor

X = tf.constant(4,shape=(1,1))
X = tf.constant([[1,2,3],[4,5,6]])   #tf.Tensor([[1 2 3] [4 5 6]], shape=(2, 3), dtype=int32)
print (X)

M = tf.ones((3,3))   #全是1
print (M)

"""
tf.Tensor(
[[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]], shape=(3, 3), dtype=float32)
"""

Y = tf.zeros((2,3))    #全是0
print (Y)
"""
tf.Tensor(
[[0. 0. 0.]
 [0. 0. 0.]], shape=(2, 3), dtype=float32)
"""

A = tf.random.normal((3,3),mean=0, stddev=1)
print (A)

B = tf.random.uniform((1,3),minval=0,maxval=1)
print (B)
#在tf.random.uniform函数中，minval和maxval是两个参数，用于指定生成的随机数的最小值和最大值。具体来说，函数将生成一个形状为(1, 3)的张量，其中每个元素都是在minval和maxval之间随机生成的浮点数。
#因此，在这个例子中，minval=0和maxval=1意味着生成的随机数将在 0 和 1 之间。如果没有指定minval和maxval，则默认值为 0 和 1。

# Indexing
x = tf.constant([0, 1, 1, 2, 3, 1, 2, 3])
print(x[:])
print(x[1:])
print(x[1:3])
print(x[::2])
print(x[::-1])

indices = tf.constant([0, 3])
x_indices = tf.gather(x, indices)

import argparse
parser = argparse.ArgumentParser(description="train")
#parser.add_argument("--train_path", type=str, default=data_dirname+"/train",help="train file")
parser.add_argument("--train_path", type=str, default="/train_sample",help="train file")
args = parser.parse_args()
print("--train_path:", args.train_path)