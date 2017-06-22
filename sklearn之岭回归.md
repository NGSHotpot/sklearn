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

### 多元线性回归中的问题


