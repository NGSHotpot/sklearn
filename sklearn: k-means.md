## 前言 

[sklearn](http://scikit-learn.org/) 是NGS Hotpot知道的最好的python machine learning package，这个最好的评价标准是：1）性能稳定；2）持续更新；3）文档完整；4）代码优美且一致性很高。NGS Hotpot通过学习sklearn的文档收获匪浅，在自己的课题中也没少用sklearn去完成一些有意思的分析。虽然近期deep learning 大火，所有方法都有往那个方向靠的趋势，但在NGS Hotpot朴素的观点中，可能并不存在完美的普适的方法，只有针对问题比较合适的方法，经典方法（经过许多人填坑的方法）依旧有其魅力。

从本篇开始，我们将一篇介绍一个sklearn集成的算法以及应用，希望大家继续共同进步。 本篇从较为经典的k-means开始。



## 算法思想   

1. 数学本质  

k-means把含有$N$个样本的数据集$X$ 切分成$k$个类别 $C$，对$C$中的每个类其中心点为$\mu_j$ , 最终使组内到中心点的平方和最小，这个可以

$ \sum_{i=0}^{N}  \sum_{j=0}^{k} min(||x_{i,j}-\mu_{j}||^2) $

2. 算法流程与伪代码   

```

随机选取k个初始点作为每个类的中心点   

当任意一点的分类结果发生改变时   

       对每个点

               对每个中心点      

                        计算点到中心点的距离

               将点分类到距离最近的中心点所在的类

       对每个类计算新的中心点

```



## 算法存在的问题  





## 使用案例   



## Reference

1. http://scikit-learn.org/stable/modules/clustering.html#clustering
