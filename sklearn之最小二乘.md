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


