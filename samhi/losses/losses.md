# 1. Loss Functions
reduce = None 表示对每一个batch size 操作输出为(batch_size,)
reduce = True 输出标量
reduction='mean' batch size的均值
reduction='sum'  batch size的和
reduce and size_average will be Deprecated
## 1.L1Loss
input 和target中每个元素的差的绝对值mean absolute error.

## 2.MSELoss
input和target中每个元素的平方差(mean squared error-squared L2 norm)

## 3.CrossEntropyLoss
```python
torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

```
交叉上损失函数,功能是将输入经过softmax激活函数后，在计算其与target的交叉熵损失。即它是由`nn.LogSoftmax()`和`nn.NLLLoss`组成.严格意义上的交叉熵损失函数应该时`NLLLoss()`
补充：交叉熵损失(cross-entropy Loss)又称对数似然损失(Log-likelihood Loss),对数损失；二分类时还可称之为(Logistic loss).交叉熵损失函数表达式为L = -sigama(y_i*log(x_i)).
在多分类任务中，经常采用softmax激活函数+交叉熵损失函数，因为**交叉熵描述了两个概率的分布差异**,然而神经网络输出时向量，并不是概率分布的形式。所以需要softmax激活函数
将一个向量进行归一化成概率分布的形式，在采用交叉熵损失函数计算loss。在回顾Pytorch的`nn.LogSoftmax()`相当与激活函数,`nn.NLLLoss()`是损失函数，将其结合，完整的是否可以叫做softmax
+交叉熵损失函数呢?
![crossEntropy](https://github.com/kadimakipp/network/raw/master/samhi/losses/images/crossEntropy.jpg)