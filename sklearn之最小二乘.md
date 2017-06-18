# sklearn之最小二乘

## 简介

本文将简单介绍使用最小二乘法来进行线性回归，并且将使用实例来说明其使用方式，最后也将用sklearn来实现。

## 线性回归介绍

若是有![equation](http://latex.codecogs.com/gif.latex?p)个自变量![equation](http://latex.codecogs.com/gif.latex?X_1,X_2,...,X_p)，且有一个因变量![equation](http://latex.codecogs.com/gif.latex?Y)。可以使用自变量对因变量建立回归模型，从而预测因变量的值，回归模型如下：

![equation](http://latex.codecogs.com/gif.latex?Y=a_0+a_1X_1+a_2X_2+...+a_pX_p+\epsilon)

在有了![equation](http://latex.codecogs.com/gif.latex?X_1,X_2,...,X_p)和![equation](http://latex.codecogs.com/gif.latex?Y)的数据之后，可以通过数据去估计![equation](http://latex.codecogs.com/gif.latex?a_0,a_1,a_2,...,a_p)的值，这些值叫对应变量的系数。最小二乘法就是用来估计这些参数的值的方法。


## 最小二乘法

对一组已知的系数![equation](http://latex.codecogs.com/gif.latex?a_0,a_1,a_2,...,a_p)，我们可以得到因变量的的估计值：

![equation](http://latex.codecogs.com/gif.latex?\hat{Y}=a_0+a_1X_1+a_2X_2+...+a_pX_p)

最小二乘的目的就是寻找到恰当的![equation](http://latex.codecogs.com/gif.latex?a_0,a_1,a_2,...,a_p)使得残差平方和最小，残差平方和计算如下：

![equation](http://latex.codecogs.com/gif.latex?\sum_{i=1}^{n}{\epsilon_{i}^{2}=\sum_{i=1}^{n}{\Vert\hat{Y_i}-Y_i\Vert_{2}^{2}})

为简便，下面使用一元线性回归来说明最小二乘法，即![equation](http://latex.codecogs.com/gif.latex?p=1)，然后所要回归的方程变为![equation](http://latex.codecogs.com/gif.latex?Y=a_0+a_1X+\epsilon)，所有估计的系数只有![equation](http://latex.codecogs.com/gif.latex?a_0,a_1)，假如总共有![equation](http://latex.codecogs.com/gif.latex?n)个样本的数据![equation](http://latex.codecogs.com/gif.latex?x_1,x_2,...,x_n)和![equation](http://latex.codecogs.com/gif.latex?y_1,y_2,...,y_n),所以残差平方和就变为了:

![equation](http://latex.codecogs.com/gif.latex?\sum_{i=1}^{n}{\epsilon_{i}^{2}=\sum_{i=1}^{n}{\Vert\hat{y_i}-y_i\Vert_{2}^{2}}=\sum_{i=1}^{n}{\Vert{a_0+a_1x_i-y_i}\Vert_{2}^{2}})

要想求得使得残差平方和最小的![equation](http://latex.codecogs.com/gif.latex?a_0,a_1)可以对![equation](http://latex.codecogs.com/gif.latex?a_0,a_1)求偏导，然后令偏导数值等于0时，此时就能取到最小值。学数学的朋友应该知道，对于某个连续可导的函数来说，若是该函数的导数在某点处取到0，那么函数值在该点达到极大值或者极小值，注意这里的极大值和极小值一个局部概念，和我们通常说的最大值和最小值是不一样的。那么为什么最小二乘里面直接求偏导为0的点就是最小值呢？可能是最大值吗？当然是不可能的，因为最小二乘中的函数衡量的是一个残差平方和，所以若是将偏差到足够大，残差平方和是可以取无穷大的，不存在最大值，所以若是存在偏导为0的点，那么该点为一个极小值点。

下面求分别对![equation](http://latex.codecogs.com/gif.latex?a_0,a_1)求偏导且令偏导等于0：

![equation](http://latex.codecogs.com/gif.latex?\frac{dy}{da_0}=2na_0+2\sum_{i=1}^{n}{x_ia_1}-2\sum_{i=1}^{n}{y_i}=0)

![equation](http://latex.codecogs.com/gif.latex?\frac{dy}{da_1}=2a_0\sum_{i=1}^{n}{x_i}+2\sum_{i=1}^{n}{x_i^2}-2\sum_{i=1}^{n}{x_iy_i}=0)

对上述结果进行整理可以到方方程组：

![equation](http://latex.codecogs.com/gif.latex?na_0+\sum_{i=1}^{n}{x_ia_1}-\sum_{i=1}^{n}{y_i}=0)

![equation](http://latex.codecogs.com/gif.latex?a_0\sum_{i=1}^{n}{x_i}+\sum_{i=1}^{n}{x_i^2}-\sum_{i=1}^{n}{x_iy_i}=0)

未知数只有![equation](http://latex.codecogs.com/gif.latex?a_0,a_1)，这个是最简单的求解二元一次方程组的问题啦，大家在初中或者小学的时候就学过了吧！然后可以得到![equation](http://latex.codecogs.com/gif.latex?a_0,a_1)的值了。最后解得结果如下：

![equation](http://latex.codecogs.com/gif.latex?a_0=\frac{\sum_{i=1}^{n}{x_i^2}\sum_{i=1}^{n}{y_i}-\sum_{i=1}^{n}{x_i}\sum_{i=1}^{n}{x_iy_i}}{n\sum_{i=1}^{n}{x_i^2}-(\sum_{i=1}^{n}{x_i})^2})

![equation](http://latex.codecogs.com/gif.latex?a_1=\frac{n\sum_{i=1}^{n}{x_iy_i}-\sum_{i=1}^{n}{x_i}\sum_{i=1}^{n}{y_i}}{n\sum_{i=1}^{n}{x_i^2}-(\sum_{i=1}^{n}{x_i})^2})

通过这两个公式，就可以根据回归的值确定系数![equation](http://latex.codecogs.com/gif.latex?a_0,a_1)了。

但是若是变量有![equation](http://latex.codecogs.com/gif.latex?p)个呢？方法是同样的，首先写出残差平方和的公式：

![equation](http://latex.codecogs.com/gif.latex?\sum_{i=1}^{n}{\epsilon_{i}^{2}=\sum_{i=1}^{n}{\Vert\hat{Y_i}-Y_i\Vert_{2}^{2}}=\sum_{i=1}^{n}{\Vert{a_0+a_1X_1+a_2X_2+...+a_pX_p-Y_i}\Vert_{2}^{2}})

然后同样对![equation](http://latex.codecogs.com/gif.latex?a_0,a_1,a_2,...,a_p)这![equation](http://latex.codecogs.com/gif.latex?p+1)个变量求偏导，并使其等于0，如此便可以得到一个![equation](http://latex.codecogs.com/gif.latex?p+1)元的方程组，解之即可。


## 应用实例

Iris数据集是很出名的机器学习数据集，总共包含三种不同的iris花的种类，其中每种花有50个数据集，每个数据集包括花萼的长度、花萼的宽度、花瓣的长度、花瓣的宽度四个数据。该数据集可以从UCI机器学习数据库下载（https://archive.ics.uci.edu/ml/datasets.html）， 这个数据集最常用作分类，但是这里我们取其中的Iris-setosa类型的数据来做回归分析。自变量![equation](http://latex.codecogs.com/gif.latex?length)为花萼的长度，因变量![equation](http://latex.codecogs.com/gif.latex?width)为花萼的宽度，建立如下线性回归模型：

![equation](http://latex.codecogs.com/gif.latex?width=a_0+a_1height+\epsilon)

然后根据刚才推出的计算回归系数的公式计算系数![equation](http://latex.codecogs.com/gif.latex?a_0,a_1)，计算代码如下：

```python
def cal_coef(x, y):
	"""
	calculate the coefficient in y = a0 + a1 * x
	"""
	n = len(x)
	square = [tmp * tmp  for tmp in x]
	multi = [x[i] * y[i] for i in range(n)]
	a0 = (sum(square)*sum(y) - sum(x)*sum(multi)) / (n*sum(square) - sum(x)**2)
	a1 = (n*sum(multi) - sum(x) * sum(y)) / (n*sum(square) - sum(x)**2)
	return a0, a1

height = ...  # read the data here, list here, [1,2,3]
weight = ... # read the data here, list here, [1, 3, 6]

a0, a1 = cal_coef(height, weight)
```
通过上面的代码可以得到![equation](http://latex.codecogs.com/gif.latex?a_0,a_1)的值分别为：

![equation](http://latex.codecogs.com/gif.latex?a_0=-0.62301)
![equation](http://latex.codecogs.com/gif.latex?a_1=0.80723)

上面的代码写起来虽然不算复杂，但是若是对于多元线性回归，就会很麻烦了。sklearn提供了linear_model的模块，可以很方便的做线性回归，对这个例子代码如下：

```python
from sklearn import linear_model

height = ...  # read the data here, list, [[2], [3], [4]]
weight = ... # read the data here, array, [1, 3, 4]

lr = linear_model.LinearRegression()
lr.fit (height, weight)

a0 lr.intercept_
a1 = lr.coef_
```
使用sklearn很方便，只需要在读入数据后，fit一下就可以得到所要求的![equation](http://latex.codecogs.com/gif.latex?a_0,a_1)了，计算结果为

![equation](http://latex.codecogs.com/gif.latex?a_0=-0.62301)
![equation](http://latex.codecogs.com/gif.latex?a_1=0.80723)

和使用自己写的函数的结果一致。

## 总结

本文主要介绍了线性回归中最小二乘法估计系数，并且给出了其在估计一元线性回归过程中的具体步骤，然后以iris数据集为例给出了最小二乘法计算系数的函数及使用sklearn来计算系数的方法，希望本文能加深大家对最小二乘及线性回归的理解。


## 参考文献

1. http://scikit-learn.org/stable/modules/linear_model.html  其中1.1.1小节


