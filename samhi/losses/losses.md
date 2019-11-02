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
交叉上损失函数，它是由`nn.LogSoftmax()`和`nn.NLLLoss`组成