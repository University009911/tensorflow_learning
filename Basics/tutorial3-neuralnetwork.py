import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


def pp():
    return print("####################################") 
#physical_devices = tf.config.list_physical_devices("GPU")
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train里面每一个element都是二维数组
print(f"x_train[5]的输出是{x_train[5]}")  #输出第五个元素
print(f"x_train[5]的长度是{len(x_train[5])}")    #x_train[5]的长度是28

pp()

x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0
print(f"x_train.shape的输出是{x_train.shape}")  #(60000, 28, 28, 1)
print(f"x_train[5]的输出是{x_train[5]}")  #输出第五个元素
print(f"x_train[5]的长度是{len(x_train[5])}")    ##输出第五个元素的长度  x_train[5]的长度是784
print(f"x_train[5].shape是{x_train[5].shape}")    #x_train[5].shape是(784,)
# import sys
# sys.exit()

"""
###/reshape(-1, 28 * 28)解释
在上面的代码中，reshape(-1, 28 * 28)是将输入的图像数据重新塑造成形状为(-1, 28 * 28)的矩阵的操作。具体来说，
它将每张 28x28 的图像转换为一个 784 维的向量，其中 784 等于 28*28，即图像的像素数量。
这样做的目的是将图像数据转换为适合训练神经网络的格式。神经网络通常需要输入一个向量，而不是一张图像，因此需要将图像数据转换为向量形式。
此外，将图像数据转换为相同大小的向量也有助于在训练过程中规范化数据，从而提高模型的准确性和泛化能力。
因此，代码中的reshape(-1, 28 * 28)语句将每张图像转换为一个 784 维的向量，以便后续训练神经网络。


在`reshape(-1, 28 * 28)`这个操作中，-1 表示在矩阵的维度中，有一个维度是未知的，需要根据输入数据的大小来确定。
具体来说，当使用`reshape`函数将一个矩阵重塑为另一个形状时，需要指定新矩阵的每一维的大小。但是，在这个操作中，有一个维度的大小被设置为-1，
这意味着它的大小将根据输入数据的大小自动计算。在上面的代码中，由于输入的图像数据是一个二维数组，因此需要将其转换为一个一维向量。在这种情况下，
使用`reshape(-1, 28 * 28)`操作可以将二维数组中的所有数据沿着一个维度展开，并将其转换为一个形状为(-1, 784)的一维向量。其中，-1 表示这个向量的长度是未知的，
需要根据输入数据的大小来确定，而 784 则表示转换后的向量中每个元素的大小，即 28 * 28 = 784。
因此，使用`reshape(-1, 28 * 28)`操作可以根据输入数据的大小自动计算新矩阵的大小，从而实现将图像数据转换为适合训练神经网络的格式的目的。
"""


"""
### /255 解释
在上面的代码中，除以 255 的操作是将图像数据归一化到范围[0, 1]内的过程。具体来说，原始的 MNist 数据集中的图像是以灰度格式存储的，
每个像素的值范围是[0, 255]，其中 0 表示黑色，255 表示白色。为了将这些图像数据转换为适合训练神经网络的格式，需要将它们归一化到范围[0, 1]内。
除以 255 的操作就是将每个像素的值除以 255，从而将其转换为浮点数，并且将其范围映射到[0, 1]之间。这样可以确保在训练神经网络时，
图像数据具有相同的范围和尺度，从而提高模型的准确性和泛化能力。因此，代码中的
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0 和 x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0
语句分别对训练集和测试集的图像数据进行了归一化操作。
"""


# Sequential API (Very convenient, not very flexible)
model = keras.Sequential(
    [
        keras.Input(shape=(28 * 28)),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(10),
    ]
)

print(model.summary())     #打印model的summary

""" 
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 512)               401920
_________________________________________________________________
dense_1 (Dense)              (None, 256)               131328
_________________________________________________________________
dense_2 (Dense)              (None, 10)                2570
================================================================= 
"""




model = keras.Sequential()
model.add(keras.Input(shape=(784)))
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(256, activation="relu", name="my_layer"))
model.add(layers.Dense(10))


print(model.summary())     #打印model的summary

""" 
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense_3 (Dense)              (None, 512)               401920
_________________________________________________________________
my_layer (Dense)             (None, 256)               131328
_________________________________________________________________
dense_4 (Dense)              (None, 10)                2570
=================================================================
Total params: 535,818
Trainable params: 535,818
Non-trainable params: 0
_________________________________________________________________
 """

# sys.exit()

# Functional API (A bit more flexible)
inputs = keras.Input(shape=(784))
x = layers.Dense(512, activation="relu", name="first_layer")(inputs)     #这一层有512个节点
x = layers.Dense(256, activation="relu", name="second_layer")(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile( 
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"],
)

""" 
这段代码是 Keras 模型的编译代码，用于定义模型的损失函数、优化器和度量指标。具体来说：
loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False) 指定了模型的损失函数为 SparseCategoricalCrossentropy 损失函数，其中 from_logits=False 表示输入的是未经 softmax 归一化的 logits 而不是概率分布。
optimizer=keras.optimizers.Adam(lr=0.001) 指定了模型的优化器为 Adam 优化器，其中 lr=0.001 表示学习率为 0.001。
metrics=["accuracy"] 指定了模型的度量指标为准确率（accuracy）。
通过使用这些参数，我们可以为模型定义合适的损失函数、优化器和度量指标，以实现最佳的训练效果。在实际应用中，你可以根据具体的任务和数据集选择不同的损失函数、优化器和度量指标。 
"""

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)   #训练model  60000份数据，每个批次32个，所有数据总共1875份   整个数据集训练5次
model.evaluate(x_test, y_test, batch_size=32, verbose=2)
