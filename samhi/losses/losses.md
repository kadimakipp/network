# 1. Loss Functions
reduce = None 表示对每一个batch size 操作输出为(batch_size,)
reduce = True 输出标量
reduction='mean' batch size的均值
reduction='sum'  batch size的和
reduce and size_average will be Deprecated

[reference](https://zhuanlan.zhihu.com/p/61379965)
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

### 4.NLLLoss
```python
torch.nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
```
功能不好用语言描述, `loss(input, class)=-input[class]`.for example 三分类任务,`input=[-1.2,2.6,0.5]`,真实标签为2(class=2),则loss为-0.5.就是对应类别输出，区一个负号!.
实际常用于多分类任务，但是input在输入NLLLoss()之前，需要对input进行log_softmax函数激活，即将input转换成概率分布的形式，并切取对数。
当带上权值，reduce = True, size_average = True, 其计算公式为:
![NLLLoss](https://github.com/kadimakipp/network/raw/master/samhi/losses/images/NLLLoss.jpg)

### 5.PoissonNLLLoss
```python
torch.nn.PoissonNLLLoss(log_input=True, full=False, size_average=None, eps=1e-08, reduce=None, reduction='mean')
```
带泊松分布的负对数似然损失,公式
```python
target~Poisson(input)loss(input, target) = input-target*log(input)+log(target!)
```

### 6.KLDivLoss
```python
torch.nn.KLDivLoss(size_average=None, reduce=None, reduction='mean')
```
功能:计算input和target之间的的KL离散度
![KLdiv](https://github.com/kadimakipp/network/raw/master/samhi/losses/images/KLdivLoss.jpg)
KL离散度又称为相对熵(Relative Entropy),用于描述两个概率分布之间的差异。计算公式(离散时):
其中p表示真实分布,q表示p的拟合分布,D(P||Q)表示当用概率分布p时,产生的信息损耗。这里的信息损耗可以理解为损失，损失越低，拟合分布q越接近真实分布p。
同时也可以从另外一个角度上观察这个公式，即计算的时p与q之间的对数差在p上的期望。特别注意,D(p||q)不等于D(q||p),其不具有对称性，因此不能称为K-L距离。
**信息熵=交叉熵-相对熵**从信息论的角度观察三者，其关系为信息熵=交叉熵-相对熵。在机器学习中，当训练数据固定，最小化相对熵D(p||q)等价于最小化交叉熵H(p,q).

