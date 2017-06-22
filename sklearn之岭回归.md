# sklearn之岭回归

## 简介

前文介绍的[sklearn之最小二乘](https://github.com/NGSHotpot/sklearn/blob/master/sklearn%E4%B9%8B%E6%9C%80%E5%B0%8F%E4%BA%8C%E4%B9%98.md)可以用来求解线性回归的回归系数，在该文中并没有提到回归过程中可能遇到的问题及解决方法，本文将阐述这部分内容。


## 线性回归最小二乘的问题

### 最小二乘存在无解的情况

之前说到最小二乘法求解线性回归系数可以转变为求解方程组的问题：

![equation](http://latex.codecogs.com/gif.latex?na_0+\sum_{i=1}^{n}{x_ia_1}-\sum_{i=1}^{n}{y_i}=0)

![equation](http://latex.codecogs.com/gif.latex?a_0\sum_{i=1}^{n}{x_i}+\sum_{i=1}^{n}{x_i^2}-\sum_{i=1}^{n}{x_iy_i}=0)

未知数只有![equation](http://latex.codecogs.com/gif.latex?a_0,a_1)，这个是最简单的求解二元一次方程组的问题啦，大家在初中或者小学的时候就学过了吧！然后可以得到![equation](http://latex.codecogs.com/gif.latex?a_0,a_1)的值了。

上面说法是对的，中小学的时候在数学课上就学过如何求解上述方程，但是也不是如我们前文所提到的那么简单，将上述方程简化成如下二元一次方程组：

![equation](http://latex.codecogs.com/gif.latex?a_1x+b_1y=c_1)

![equation](http://latex.codecogs.com/gif.latex?a_2x+b_2y=c_2)

那么由上面方程组可以得到：


![equation](http://latex.codecogs.com/gif.latex?D=\begin{vmatrix}a_1&b_1\\\\a_2&b_2\end{vmatrix})

![equation](http://latex.codecogs.com/gif.latex?D_x=\begin{vmatrix}c_1&b_1\\\\c_2&b_2\end{vmatrix})

![equation](http://latex.codecogs.com/gif.latex?D_y=\begin{vmatrix}a_1&c_1\\\\a_2&c_2\end{vmatrix})

由已知克拉默定理可以得到，若是![equation](http://latex.codecogs.com/gif.latex?D\ne0)时，方程组有唯一解，其解分别为：

![equation](http://latex.codecogs.com/gif.latex?x=\frac{D_x}{D})

![equation](http://latex.codecogs.com/gif.latex?y=\frac{D_y}{D})

这个时候我们就很开心，线性回归直接用最小二乘就做好了。

但是若是![equation](http://latex.codecogs.com/gif.latex?D=0)呢？这个时候就麻烦了，方程组要么无解要么有无数解。任何一种情况都会导致使用最小二乘估计回归系数失效（当然在真实数据中，这种情况发生得不多）。

再说上面有解的情况，求解的过程需要计算矩阵的行列式，若是矩阵的中某一个值有较小的改变就会导致行列式的有较大的改变，这显然对于实际应用是有极大影响的。这种情况是可能存在的。


### 多元线性回归中的问题

说一个多元线性回归存在的问题，当使用最小二乘计算多元线性回归系数时，若是多个自变量间由较大的相关性，此时方差膨胀因子会增大。此时回归系数估计得会很不准确，做检验的话很难显著。

既然使用最小二乘估计线性回归存在这些问题，所以有了岭回归来对这些问题进行改善。

## 岭回归介绍

最小二乘是要使得下面的残差平方和最小，

![equation](http://latex.codecogs.com/gif.latex?\sum_{i=1}^{n}{\epsilon_{i}^{2}=\sum_{i=1}^{n}{\Vert\hat{Y_i}-Y_i\Vert_{2}^{2}})

二岭回归在最小二乘的残差平方的和的基础上加上一个和系数成比例的惩罚项，如下所示

![equation](http://latex.codecogs.com/gif.latex?\sum_{i=1}^{n}{\epsilon_{i}^{2}=\sum_{i=1}^{n}{\Vert\hat{Y_i}-Y_i\Vert_{2}^{2}}+\sum_{i=1}^{p}{\alpha\Vert{a_i}\Vert^{2}})

加上惩罚项之后有两个作用，第一是会使得回归的性能没有不加好，即回归的方差解释度会下降，第二是回归的系数更稳当，也就是更容易显著。当自变量间有较强相关性或者数据比较异常的时候可以使用岭回归(注意，惩罚项只加变量的系数平方，截距项![equation](http://latex.codecogs.com/gif.latex?a_0)不加)。

然后可以使用和最小二乘相同的方式进行求解。先求导，然后使得导数数值为0。

![equation](http://latex.codecogs.com/gif.latex?\frac{dy}{da_0}=2na_0+2\sum_{i=1}^{n}{x_ia_1}-2\sum_{i=1}^{n}{y_i}=0)

![equation](http://latex.codecogs.com/gif.latex?\frac{dy}{da_1}=2a_0\sum_{i=1}^{n}{x_i}+2\sum_{i=1}^{n}{x_i^2a_1}-2\sum_{i=1}^{n}{x_iy_i}+2\alpha{a_1}=0)

对上述结果进行整理可以得到方程组：

![equation](http://latex.codecogs.com/gif.latex?na_0+\sum_{i=1}^{n}{x_ia_1}-\sum_{i=1}^{n}{y_i}=0)

![equation](http://latex.codecogs.com/gif.latex?a_0\sum_{i=1}^{n}{x_i}+(\sum_{i=1}^{n}{x_i^2}+\alpha)a_1-\sum_{i=1}^{n}{x_iy_i}=0)

然后就可以求解得到想要的结果啦

![equation](http://latex.codecogs.com/gif.latex?a_0=\frac{(\sum_{i=1}^{n}{x_i^2+\alpha)}\sum_{i=1}^{n}{y_i}-\sum_{i=1}^{n}{x_i}\sum_{i=1}^{n}{x_iy_i}}{n(\sum_{i=1}^{n}{x_i^2}+\alpha)-(\sum_{i=1}^{n}{x_i})^2})

![equation](http://latex.codecogs.com/gif.latex?a_1=\frac{n\sum_{i=1}^{n}{x_iy_i}-\sum_{i=1}^{n}{x_i}\sum_{i=1}^{n}{y_i}}{n(\sum_{i=1}^{n}{x_i^2}+\alpha)-(\sum_{i=1}^{n}{x_i})^2})

## 应用实例
本文还是使用Iris数据集进行测试说明：Iris数据集是很出名的机器学习数据集，总共包含三种不同的iris花的种类，其中每种花有50个数据集，每个数据集包括花萼的长度、花萼的宽度、花瓣的长度、花瓣的宽度四个数据。该数据集可以从[UCI机器学习数据库下载](https://archive.ics.uci.edu/ml/index.php)，同时我也将用到的这个数据放到了本目录data文件夹下，这个数据集最常用作分类，但是这里我们取其中的Iris-setosa类型的数据来做回归分析。自变量![equation](http://latex.codecogs.com/gif.latex?length)为花萼的长度，因变量![equation](http://latex.codecogs.com/gif.latex?width)为花萼的宽度，建立如下线性回归模型：

![equation](http://latex.codecogs.com/gif.latex?width=a_0+a_1height+\epsilon)

我们使用岭回归的方式来求解这个问题，
```python
from sklearn import linear_model

def readData(infile):
    """
    read the iris data
    """
    data = map(lambda x:x.strip().split(","), open(infile).readlines())
    height = [float(x[0]) for x in data]
    width = [float(x[1]) for x in data]
    return height, width
    
def my_ridge(x, y, alpha):
    """
    ridge regression for simple linear regresion
    """
    n = len(x)
    square = [tmp * tmp  for tmp in x]
    multi = [x[i] * y[i] for i in range(n)]
    a0 = ((sum(square)+alpha)*sum(y) - sum(x)*sum(multi)) / ((n)*(sum(square)+alpha) - sum(x)**2)
    a1 = ((n)*sum(multi) - sum(x) * sum(y)) / ((n)*(sum(square)+alpha) - sum(x)**2)
    return a0, a1

def sklearn_ridge(x, y, alpha):
    """
    input x should be 2D array, such as [[1], [2], ...]
    """
    clf = linear_model.Ridge(alpha=alpha)
    clf.fit(x, y)
    return clf.intercept_, clf.coef_
    
height, width = readData(""data/iris_test.in"")
height2 = [[x] for x in height]
alpha = 0.5
a0, a1 = my_ridge(height, width, alpha)
b0, b1 = sklearn_ridge(height2, width, alpha)
print "the results for my ridge: a0=",a0, "a1=",a1
print "the results for sklearn ridge: a0=", b0, "a1=", b1
```
执行上述代码可以得到两种计算的结果是相同的。

![equation](http://latex.codecogs.com/gif.latex?a_0=-0.31632615889)

![equation](http://latex.codecogs.com/gif.latex?a_1=0.74597007)

## 总结

本文主要介绍了岭回归的方法进行线性回归，主要讲解和代码都是对简单线性回归进行的，对于多元的情况是使用矩阵计算来完成的。后面我们会慢慢介绍到。

## 参考文献

1. http://scikit-learn.org/stable/modules/linear_model.html#ridge-regression 1.1.2节
