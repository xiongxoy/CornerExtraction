基于方向向量聚类的多边形定点提取# 问题意义
精确定位多边形的顶点坐标是一个很有实际价值的问题。例如，在进行车牌识别或者是纸币识别的时候，需要将倾斜的车牌和纸币旋转到统一的相位和大小，后续的模式识别算法才能更好地发挥作用。另外，提取多边形的顶点，还能用于对多边形的压缩编码，极大的减少传输纯多边形图像所需的空间。
因为矩形在经过投影变换之后，变成了四边形，如何提取四边形的顶点，就成了一个很重要的问题。
## critical point的问题意义The work on the detection of dominant points started from the research of Attneave who proposed that the local maximum curvature points on a curve have a rich infomation conent and are sufficent to characterize this curve. A method for detection of dominent points can lead to a good representation of a planar shape at different resolutions. 
In addition, a representation of a planar shape based on dominant points has some advantages.
Firstly, it enables a high data reduction. Secondly, this representation concetrates on principal features of the shape, so it is efficient for shape matching, feature extraction or decomposition of the curve into meaningful parts. Therefore, these points have a critical role in curve approximation, shape matching and image matching.They also lead to some applications in other domains of machine vision such as vector data compression. Starting from Attneave's work, there are many existing methods for dominant points detection. Concerning this problem, several problems in this topic have been identified: evaluation, number of parameters, selection of starting point, mutil-scale, working with noisy curvets, ... .
In general, we can classify these methods into two groups. The first one contains direct methods that determine dominant points such as high curvature value points by using curvature-based significant measures, or using alternative significant measures such as k-cosine, region of support (ROS). Resenfield and Johnston used cosine of the angle between the arcs of length k on each side of a point (termed k-cosine) as curvature-based significant measure. Teh and Chin proposed a non-parametric method for detecting dominant points. They used ROS as significant measure that is determined at each point p_i thanks to its local properties (chrod p_{i-k}{i+k}) and perpendicular distance from p_i to this chord). The dominant points are finally detected by a non-maximum suppression process. They also concluded in this work that the performance of a dominant point detection algorithm depends not only on the accuracy of significant measure, but also on the precise determination of ROS. Marji and Siy determined ROS by using intergral square error. They selected end points of support arm as dominant points upon the frequency of their selection. Other algorithms expoit the rule, iteration on neighboring pixels. Sarkar proposed a simple non-parametric method for detecting dominant points from a chain-code boundary. This algorithm examines the differential chain-code of the boundary points sequentially and confirms the significant points based on pure chain code manipulation. Cronin assigned a special code the each boundary point based on the Freeman chain code applied for its two immediate neigbors. 
The inital dominant points are the set of points with non-zero differential chain code. An elimination process is followed where the boundary is searched exhaustively for predefied sequences to elminate them from initial dominant set.
The indirect methods are often based on polygonal approximation of the curve, the dominant points are dedeced after doing this step. In these methods, the dominant points are detected as the vertex of approximated polygons. In addition, we can divide polygonal approximation methods into three principal approaches: 
1. sequential approach

 Ray and Ray determined the longest possible line segments with minimum error. Kolesnikov proposed a sub-optimal algorithm for polygon approximation of closed curves based on the corresponing optimal dynamic programming algorithm for open curves.
 
 Aoyama used a linear scan to evaluate  error conditions, if the conditions are not satisfied, a new segment search is started. The problem of this is that sometimes the nodes do not correspond to the corners because a new vector is defined only when the conditions are violated.
  2. split and merge approach


3. heuristic search one# 相关方法## 霍夫变换一种方式是使用霍夫变换，即首先找到图像中所有的直线，然后求其交点。该方法的缺点是，对噪声敏感，只能处理比较理想的情况，严重依赖参数。## 计算几何另外一种思路是使用计算几何的方法求解。给定一个点集，现存的计算几何方法可以很高效找到其minimum enclosing rectangle以及minimum enclosing k-gon。这类方法的问题是，它只能处理凸多边形的情况，而且不能很好的处理离群点。

minimum enclosing rectangle的问题在于，它不能处理一般的四边形。minimum enclosing k-gon的问题在于概念复杂，没有很好的公开实现，对噪声敏感，不能处理凹多边形。

## critical point
不是精确定位顶点的方法，信息的压缩率不够。不精确的定位，也导致了它不能用于某些场景。虽然通过确定max-width可以一定程度上解决这个问题，但是不能直接满足最终目的，当数据量很多，如有很多车牌需要矫正的时候，无法完全自动化的完成顶点的定位，会直接导致算无法使用。# 本文方法本文从数据分析（data clustering）的角度来处理这个问题，以临接两点之间构成的方向向量为特征，使用聚类来确每个点所属的直线，然后拟合直线，并求其交点来确定多边形的顶点。本文将该方法用于仿射变换的正则化，通过校正倾斜的图案，表明本方法确实可行。本方法的核心实质上是直线提取，但和霍夫变换相比，更具有针对性。本方法将重点放在了顶点所属直线的确定上，结合问题的特点，提出了高效可行的算法。

## 凸多边形
凸多边形的情况比较好处理，可以先求轮廓的凸包，因为凸包相邻点的单位方向向量，在单位圆上，总是相邻的，而且同一条边上点所属的方向向量，在单位圆上也是相邻的，因此可以在单位圆上进行kmeans聚类，从而得到各个点所从属的直线，在通过直线拟合即可求出多边形的各条边，进而求出顶点。

### 关于聚类算法选取

有两种选择，kmeans或者spectral clustering。

kmeans是一种经典的聚类算法，但是它对cluster的形状有要求，当cluster接近高斯分布的时候，它的表现最佳。本问题中，各直线所包含的点，都是相临接的，因此kmeans算法能取得很好的效果。

spectral clustering是一种较为新的聚类算法，该算法的特点是，不依赖cluster的形状，本质上是一种Nearest Neighbor聚类。由spectral clustering的特性可知，kmeans能处理的聚类任务，spectral clustering也能很好的处理。因此，如果可以把多边形的各条边映射成高维空间中不相邻的点云，spectral clustering就可以用于提取凹多边形的各条边。不过，实际中的轮廓，各邻接点之间的方向向量，并不规则，其在单位圆上的分布存在交叉，因此spectral clustering并不能很好的处理这个问题。

#### spectral clustering是否可以用于凹多边形的顶点提取的实验


### K-means的描述
Let X = {x_i}, i=1,\dots,n be the set of n d-dimensional points to be clustered into a set of K clusters, C = {c_k, k = 1, \dots, K}. K-means algorithm finds a partition such that the squared error between the empirical mean of a cluster and the points in the cluster is minimized. Let \mu_k be the mean of cluster c_k. The squared error between \mu_k and the points in cluster c_k is defined as

 $$ J(c_k) = \sum_{x_i\in c_k}||x_i - \mu_k||^2 $$
 
The goal of K-means is to minimize the sum of the squared error over all K clusters.

 $$ J(C) = \sum_{ k=1}^K \sum_{x_i \in c_k} || x_i - \mu_k|| ^2 $$
 
 K-means starts with an initial partition with K clusters and assign patterns to clusters so as to reduce the squared error. Since the squared error always decreases with an increase in the number of clusters K (with J(C) = 0 when K = n), it can be mini- mized only for a fixed number of clusters. The main steps of K- means algorithm are as follows:
 
1. Select an initial partition with K clusters; repeat steps 2 and 3 until cluster membership stabilizes.2. Generate a new partition by assigning each pattern to its closest cluster center.3. Compute new cluster centers.
## 凹多边形
求凹多边行的顶点的困难之处在于，其边没有很好的性质，方向向量会发生变化，目前没有很好的求取其精确顶点的算法。

本文使用RANSAC算法的思想，求解该问题
1. 随机选取一个点，以该点为中心，选取一个ROS（Region of Support）
2. 在ROS的范围内，进行直线拟合
3. 删除直线范围内的点
4. 对于K边形，重复1-3，K次，得到K条直线，记录点的覆盖率
5. 重复1-4，N次，选出覆盖率最高的直线组。
6. 用直线求出该凹多边形的顶点

### RANSAC 的描述

The RANSAC algorithm is a 
### ROS的确定# 实验
## 车牌的矫正

先前的critical point提取算法会提取contour中不平滑的部分，由此得到的点并不一定是多边形的顶点，特别是当contour中噪声比较多的时候，critical point的数量会很多。而在某些情况下，需要精确定位多边形的顶点，例如对车牌进行deskew的时候，需要根据顶点的对应关系，来进行perspective projection的矫正。

## 五角星顶点的提取

提取五角星的顶点，作为展示。
