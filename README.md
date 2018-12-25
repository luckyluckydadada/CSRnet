# CSRNet-keras
CSRNet模型使用卷积神经网络将输入图像映射到其各自的密度图。
该模型不使用任何全连接层，因此输入图像的大小是可变的。
模型从大量不同的数据中学习，并且考虑到图像分辨率，没有信息丢失。
在预测时不需要重新、调整图像大小。
模型体系结构使得输入图像为（x，y，3），输出是尺寸（x / 8，y / 8,1）的desnity图。 
通过对密度图上的值求和为预测的总人数。
[CSRNet原理博客](https://blog.csdn.net/weixin_41965898/article/details/85246709)

## Requirements 
1. Keras : 2.2.2
2. Tensorflow : 1.9.0
3. Scipy : 1.1.0
4. Numpy : 1.14.3
5. Pillow(PIL) : 5.1.0
6. OpenCV : 3.4.1

## 数据集
使用的数据集是ShanghaiTech数据集： [Drive Link](https://drive.google.com/file/d/16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI/view)
数据集分为A和B两部分。(已经下载到shanghai_data/)
A部分由人群密度高的图像组成。 
B部分由稀疏人群场景图像组成。

## 数据预处理
在数据预处理中，主要目标是将ShanghaiTech数据集提供的图像转换为密度图。
对于给定图像，数据集提供了由该图像中的头部注释组成的稀疏矩阵。
通过高斯滤波器将该稀疏矩阵转换为2D密度图。
密度图中所有单元格的总和导致该特定图像中的实际人数。
请参阅Preprocess.ipynb。

## 数据预处理数学解释
给定一组头部坐标，我们的任务是将其转换为密度图。
构建头部坐标的kdtree（kdtree是允许快速计算K最近邻居的数据结构）。
找到每个头部的平均距离，其中K（K为4）头部坐标中最近的头部。如本文作者所建议的那样，将该值乘以因子0.3。
将此值设为sigma并使用2D高斯滤波器进行卷积。

## 模型


### 模型架构分为前端和后端两部分。
前端由VGG16模型的13个预训练层组成（10个卷积层和3个MaxPooling层），未采用VGG16的完全连接层。
后端包括扩张卷积层（空洞卷积）。
根据CSRNet论文的建议，实验发现获得最大准确度的膨胀率为2。

### 代码中提供了BN功能。
由于VGG16没有任何BN层，我们构建了一个定制的VGG16模型，并将预先训练的VGG16权重移植到该模型中。

### 可变尺寸输入
在keras中，难以训练输入图像的大小可变的模型，Keras不允许在同一批次中训练可变大小的输入。
解决此问题的一种方法是组合具有相同图像尺寸的所有图像并将它们作为批次进行训练，ShanghaiTech数据集不包含许多具有相同图像大小的图像，因此方法不可行。
另一种方法是独立地训练每个图像并在所有图像上运行循环（随机梯度下降，batch_size=1），这种方法在内存使用和计算时间方面效率不高，因此方法不可行。
因此，我们提出一种方法在keras中构建了自定义数据生成器，以有效地训练可变大小的图像，使用数据生成器，可以实现高效的内存使用，并且培训时间大幅缩短。
论文还指定了图像裁剪作为数据增强的一部分，但是，Pytorch的版本实现不会在训练时使用图像裁剪。
因此，我们提供了一个函数preprocess_input（），可以在image_generator（）中使用它来添加裁剪功能。
我们已经训练了没有剪裁图像的模型。

## 训练  
数据集A和B的两个部分在两个单独的模型上进行训练，这两个模型都训练了200 epochs。
其他超参数保持与CSRNet论文和pytorch实现中指定的相同。
请参阅Model.ipynb。

## 预测
模型A在密集的人群中表现得非常好，而模型B在稀疏的人群中表现得非常好。
两种模型生成的密度图都足够精确，可以描绘人群密度的变化。
请参阅Inference.ipynb以生成推理。
下面给出了从ShanghaiTech数据集中提供的测试集中获取的实际图像的结果。

实际图片：

<img src="https://github.com/luckyluckydadada/CSRnet/blob/master/image/0.9734313601422697.png" width="480">

生成的密度图 : 

<img src="https://github.com/luckyluckydadada/CSRnet/blob/master/image/0.4779446876370941.png" width="480">

实际人数: 258
预测人数: 232

## 结果

下面给出了我们模型产生的MAE误差之间的比较。

| Dataset | MAE |  
| ------------------- | ------------- |
|ShanghaiTech part A | 65.92 | 
|ShanghaiTech part B | 11.01 |

## 参考
https://github.com/Neerajj9/CSRNet-keras

