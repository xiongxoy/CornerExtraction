基于方向向量聚类的多边形定点提取




minimum enclosing rectangle的问题在于，它不能处理一般的四边形。minimum enclosing k-gon的问题在于概念复杂，没有很好的公开实现，对噪声敏感，不能处理凹多边形。

## critical point
不是精确定位顶点的方法，信息的压缩率不够。不精确的定位，也导致了它不能用于某些场景。





有两种选择，kmeans或者spectral clustering。

kmeans是一种经典的聚类算法，但是它对cluster的形状有要求，当cluster接近高斯分布的时候，它的表现最佳。本问题中，各直线所包含的点，都是相临接的，因此kmeans算法能取得很好的效果。

spectral clustering是一种较为新的聚类算法，该算法的特点是，不依赖cluster的形状，本质上是一种Nearest Neighbor聚类。由spectral clustering的特性可知，kmeans能处理的聚类任务，spectral clustering也能很好的处理。spectral clustering相比kmean的另外一种优势是（还需要实验）

#### spectral clustering是否可以用于凹多边形的顶点提取呢？？？












### RANSAC 的描述



先前的critical point提取算法会提取contour中不平滑的部分，由此得到的点并不一定是多边形的顶点，特别是当contour中噪声比较多的时候，critical point的数量会很多。而在某些情况下，需要精确定位多边形的顶点，例如对车牌进行deskew的时候，需要根据顶点的对应关系，来进行perspective projection的矫正。

## 五角星顶点的提取

提取五角星的顶点，作为展示。
