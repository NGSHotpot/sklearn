## 前言

用不同的聚类方法，甚至是同样的聚类方法不同的参数，往往会得到不同的结果，最简单的例如我们介绍过的k-means的k的设定，然后怎么判断k是否合适呢？此外，假设自己写了一个算法，提升了效率但是能得到和原算法一样的结果，如何来证明这件事呢？sklearn的文档中关于聚类性能评估的各种指标很贴心的把这些都考虑了。NGS Hotpot这里挑选一下用过的觉得很好的指标记录一下。 

## 有真实的分类结果(ground truth)   
1.  Adjusted Rand Index      

对于两个分类结果![equation](http://latex.codecogs.com/gif.latex?X={x_1,...x_n},Y={y_1,...,y_m}), 例如**X**是真实结果，数据可以分成**n** 类，而**Y**是聚类结果，分成了**m**类，令**a**为在**X**和**Y**中都被分到一个类的点对的数量，令**b**为在**X**和**Y**中都不被分到一个类的点对的数量， 则rand index为![equation](http://latex.codecogs.com/gif.latex?RI=\frac{a+b}{C_{n}^{2}}$)

而修正后的rand index如下, 其具体计算方法可见wiki上的[列联表公式](https://en.wikipedia.org/wiki/Rand_index):    
![equation](http://latex.codecogs.com/gif.latex?ARI=\frac{RI-Expected_{RI}}{max(RI)-Expected_{RI}})

ARI的取值范围为[-1,1]， 聚类结果完全随机则ARI=0, 聚类结果完美之极，则ARI=1.0 。这个方法应该特别适合写了一个算法然后有合适的模拟数据用来评估算法效果。 

1. Fowlkes-Mallows index  

定义非常简单明了 ![equation](http://latex.codecogs.com/gif.latex?FMI=\frac{TP}{\sqrt{(TP+FP)(TP+FN)}}) , 其中**TP**是true positive, **FP** 是false positive, **FN**是false negative。这样这个值的范围也是0-1, 1代表完美聚类。

## 没有真实的分类结果
1. Silhouette Coefficient   

这在常规的k-means中选择k用的比较多，跑很多次k然后画出Sihouette Coefficient的均值情况，选个这个值最高的k作为最终的k，例如下面的示例。

![](http://scikit-learn.org/stable/_images/sphx_glr_plot_kmeans_silhouette_analysis_003.png)

令**a**为同一个类中的一个点到其他点的平均距离，令**b**为一个类中的点到其最近的其他类的所有点的平均距离， 则![equation](http://latex.codecogs.com/gif.latex?SC=\frac{b-a}{max(a,b)})， **SC**取值范围为-1到1， -1则是非常糟糕的聚类，0则代表分类结果其实含有已经被合并的类，1则代表完美。个人觉得，这个指标如果得到的结果全是负的话，其实可以考虑放弃k-means了。

1. Calinski-Harabaz Index

这个指标也可以在没有ground truth的时候评估聚类结果的好坏，值越高表明越高，简单理解可以是类间平均距离除以类内距离的标准化后的结果， 比较难过的是，没有一个取值范围。

## Reference 
1. http://scikit-learn.org/stable/modules/clustering.html
