不明白的算法：

**Approximate K Nearest Neighbor Search (AKNNS)**

NonGraphBased ANNS Methods includes tree-based method hashing-based methods and quantization-based methods：

**KD-tree R*-tree VA-file LSH and PQ**

nongraph methods的精度没有graph的好，是因为nongraph主要集中在划分空间和在子空间中检索结果，所以邻居区域没有被有效率的浏览。当维度增加的时候，就更不能很好的精度了。

graph-based anns methods

including：Delaunay Graphs

kNN graph ： use hashing and Randomized KD-trees to provide better starting positions for Algorithm 1 on the kNN graph

DPG ： built upon Knn graph， edge selection strategy

RNG： edge selection strategy， reduce out-degree。add edges to RNG called MSNET

Navigable Small-World Networks：

Randomized Neighborhood Graphs