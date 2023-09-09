"""
https://www.52txr.cn/2022/metricslossacc.html    代码来源  可以跟着flow\tensorflow_learning\Basics\tutorial3-neuralnetwork.py比较着看


https://www.youtube.com/watch?v=w8yWXqWQYmU    Building a neural network FROM SCRATCH (no Tensorflow/Pytorch, just numpy & math)
https://www.kaggle.com/code/wwsalmon/simple-mnist-nn-from-scratch-numpy-no-tf-keras/notebook  code
"""


import tensorflow as tf
(train_image, train_labels),_ = tf.keras.datasets.mnist.load_data()

# 对图片进行预处理
train_image = tf.expand_dims(train_image,-1) # 扩增维度
#print(train_image.shape)  #(60000, 28, 28, 1)
#print(train_image[0])    #输出第一个element


# 计算梯度必须使用浮点型
# 归一化
train_image = tf.cast(train_image/255, tf.float32)
#print(train_image.shape)  #(60000, 28, 28, 1)
#print(train_image[0])    #输出第一个element
#print(train_image[0].shape)    #输出第一个element的维度   (28, 28, 1)


# 对label进行处理
train_labels = tf.cast(train_labels, tf.int64)

# print(train_labels.shape)  #(60000,)
# print(train_labels[0])    #输出第一个element  tf.Tensor(5, shape=(), dtype=int64)
# print(train_labels[0].shape)    #输出第一个element的维度 ()

dataset = tf.data.Dataset.from_tensor_slices((train_image, train_labels))

# 连续、批次
dataset = dataset.shuffle(10000).batch(32)

"""
这段代码是使用 TensorFlow 库中的tf.data.Dataset模块创建一个数据集，并对数据进行随机打乱和批处理。
首先，使用tf.data.Dataset.from_tensor_slices函数将训练图像和训练标签转换为tf.data.Dataset对象。
该函数接受一个元组作为参数，其中第一个元素是训练图像，第二个元素是训练标签。
接下来，使用dataset.shuffle(10000)函数对数据集进行随机打乱。其中参数 10000 表示打乱的次数，可以根据需要进行调整。
最后，使用dataset.batch(32)函数对数据集进行批处理。其中参数 32 表示每批数据的大小，即每次取出 32 个样本进行处理。
这样做的目的是为了提高计算效率，减少内存占用。
需要注意的是，tf.data.Dataset模块是 TensorFlow 2.0 中的新特性，它提供了一种高效的数据处理方式，可以提高模型的训练效率和精度。
"""

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16,[3,3],activation="relu",input_shape=(None,None,1)),
    tf.keras.layers.Conv2D(32,[3,3],activation="relu"),
    tf.keras.layers.GlobalMaxPool2D(),
    tf.keras.layers.Dense(10)
])

""" 
这段代码定义了一个使用卷积神经网络（CNN）架构的 Keras 模型。
模型的输入是一个三维张量，形状为（None, None, 1），其中 None 表示输入的图像大小可以是任意的。模型的输出是一个十维的向量，表示预测的结果。
模型由以下几个层组成：
tf.keras.layers.Conv2D：这是一个卷积层，使用 16 个大小为 3x3 的卷积核，激活函数为 ReLU。
tf.keras.layers.Conv2D：这是另一个卷积层，使用 32 个大小为 3x3 的卷积核，激活函数为 ReLU。
tf.keras.layers.GlobalMaxPool2D：这是一个全局最大池化层，用于减少特征图的维数。
tf.keras.layers.Dense：这是一个全连接层，使用 10 个神经元，用于对特征图进行分类。
该模型可以用于图像分类任务，例如手写数字识别、图像分类等。
 """



print (model.summary())

""" 
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, None, None, 16)    160
_________________________________________________________________
conv2d_1 (Conv2D)            (None, None, None, 32)    4640
_________________________________________________________________
global_max_pooling2d (Global (None, 32)                0
_________________________________________________________________
dense (Dense)                (None, 10)                330
=================================================================
Total params: 5,130
Trainable params: 5,130
Non-trainable params: 0
_________________________________________________________________
"""

# import sys
# sys.exit()


optimizer = tf.keras.optimizers.Adam()   # 优化器，默认Adam参数

loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)   #多分类常用的损失函数

train_loss = tf.keras.metrics.Mean("train_loss")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')    
#定义了两个tf.keras.metrics，分别是train_loss和train_accuracy。这两个指标用于记录训练过程中的损失和准确率。


 
def train_step(model,images,labels):
    with tf.GradientTape() as t:
        pred = model(images)
        loss_step = loss_func(labels,pred)
    grads = t.gradient(loss_step, model.trainable_variables)
    optimizer.apply_gradients(zip(grads,model.trainable_variables))
    train_loss(loss_step)
    train_accuracy(labels,pred)

"""    
具体地，第一行代码使用tf.GradientTape计算损失函数的梯度，其中t是一个tf.GradientTape对象。然后，使用t.gradient
函数计算损失函数相对于模型中所有可训练变量的梯度。这些梯度存储在变量grads中。
第二行代码使用optimizer优化器应用梯度更新模型的权重参数。optimizer.apply_gradients函数接收两个参数：
梯度变量和模型中的可训练变量。这里，我们使用zip函数将grads和模型中的可训练变量打包在一起，以便optimizer可以一次性更新所有的权重参数。
通过不断地执行训练步骤，优化器将根据梯度更新模型的权重参数，以最小化损失函数，提高模型的性能。 
"""

def train():
    for epoch in range(10):     # 训练10个epoch
        for (batch,(images,labels)) in enumerate(dataset):
            train_step(model,images,labels)
        print("Epoch{} loss is {},accuracy is {}".format(epoch,
                                                         train_loss.result(),
                                                         train_accuracy.result()))
        # 对于下一步进行重置
        train_loss.reset_states()
        train_accuracy.reset_states()

train()        