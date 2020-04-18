

# EFANNA : An Extremely Fast Approximate Nearest Neighbor Search Algorithm Based on kNN Graph

## 0 Abstract

这篇文章是图方法，传统方法如层级结构（树）随着维度增加性能会下降，hash-based方法在实际中又缺乏效率。所以graph-based方法最近获得很多关注。本文基于的思想是“The main idea is that a neighbor of a neighbor is also likely to be a neighbor, which we refer as NN-expansion. These methods construct a k-nearest neighbor (kNN) graph offline”。在线查找时，这种方法查找到query节点的候选neighbors，比如random selection，然后check这些neighbors的neighbors选出closer

 ones。不过这个方法也不是毫无问题，也会有两个问题，一个是会收敛到局部最优值里面，另一个是build knn graph很耗时。所以本文提出了EFANNA based on knn graph，这种方法很好的结合了层级结构的有点和最近邻图的方法。是目前最快的算法在近最近邻图构建和近最近邻搜索。

## 1 Introduction

sparse data（例如文档查找）可以用advanced index（例如倒排索引），但是如果data有dense feature，找nearest neighbor的消耗时间是O(N)，N是database的节点数。如果数据库很大的时候，会很耗时间，所以ANN就被提出了，approximate nearest neighbor。

ANN有两种方式的方法：一种基于层级结构的方法，比如KD-tree，K-means treee，Randomized kd-tree等。当数据维度相对比较低的时候，性能很very well。但是当维度增加，性能会急剧下降。第二种方法是hashing based 方法，例如LSH，spectral hashing，iterative Quantization等等。



论文引用[33] 一个detailed survey on various hashing methods。**

hashing based methods主要思想是：“These methods generate **binary codes** for high dimensional real vectors while try to **preserve the similar**

**ity among original real vectors**. Thus, **all the real vectors fall into different hashing buckets** (with different binary codes). Ideally, if **neighbor vectors fall into the same bucket or the nearby buckets** (measured by the hamming distance of two binary codes), the hashing based methods can efficiently retrieve the nearest neighbors of a query point”。该方法如果要保证一个high recall，就需要扩大搜索，需要检查很多个hashing bucket。这样会造成性能降低。**论文引用[22]有详细分析。**

图方法的核心就如abstract里面说的，邻居的邻居很大可能也是邻居，NN-expansion。这些方法可以建立一个knn图。不过NN很容易陷入局部最优，导致一个low recall。可以通过hashing based methods去替代random selection来做初始化，这种方法叫**IEH（iterative expanding hashing）**。另一个用图方法的问题是创建图的过程中，很耗时间。也有很多方法去降低复杂度的，例如[4], [9], [32],[28]，不过当大数据时，都不够理想。所以，目前考虑是不去建立一个exact knn graph，而是建立一个近似的knn，“**[12] proposed NN-descent** to efficiently build an approximate kNN graph. **[8], [16], [37], [31]** try to build an approximate kNN graph in a divide-and conquer manner”。总结来说，这些方法总共有三个阶段：

1、将数据分subsets，多次；2、在subsets中do brute force search，并且得到很多有overlap的子图；3、融合子图并做调整。

不过缺陷在于，“there are no formal study on how the performance of graph based search methods will be affected if one uses an approximated kNN graph instead of an exact kNN graph”。我理解就是没有系统的科学解释吧？

所以作者提出了novel graph-based approximate nearest neighbor search framework。EFANNA index包括两部分：“the **multiple randomized hierarchical structures** (e.g., randomized truncated KD-tree) and an **approximate k-nearest neighbor graph**.”

离线构建图流程：dataset to subsets in fast and hierarchy way 生成多个rnd hierarchy structure => 生成appr knn graph along the structure（这样就不是计算所有点，而是利用层级结构来寻找最近可能的邻居了，自底向上查找） => refine graph 与NN-descent一样，用local join，sampling 和early termination。**细节[12]**。

在线查找流程：先通过层级结构找到query的候选邻居；refine results在appr knn graph上using NN-expansion。

理解：efanna的主要目的是高速有效的建图结构，以便于索引查找，但其实也可以用于其他ml场景上去建图。而且采用这种算法建图，及时建图的acc精度不高，在graph-based ann search方法中，仍然表现很优秀。

## 2 Related Work

与ANNS有关的index结构和方法，分为图方法和非图方法

KD-tree [13]，new hierarchical structure based methods [30], [7], [27]

Randomized KD-tree [30] and Kmeans tree [27]

Hashing based algorithms [17], [34]

Locality Sensitive Hashing (LSH)[17]

**Both the hashing based methods and tree based methods have the same goal. They expect to put neighbors into the same hashing bucket (or node).**

Graph Nearest neighbor Search (GNNS) [18]

Iterative Expanding Hashing (IEH) [22]

**all the graph based methods need a kNN graph as the index structure，how to build a kNN graph efficiently became a crucial problem**

[12] proposed an NN-descent algorithm

## 3 EFANNA ALGORITHMS FOR ANN SEARCH

分为offline和online两部分

### 3.1 EFANNA index

为NN-expansion提供一个好的初始化。“The **multiple hierarchical structures** is used for initialization and the **approximate kNN graph is used** for NN-expansion.”

在论文中，使用的是**Randomized truncated K-D tree**。eg:Based on this randomized truncated KD-tree, we can get the initial neighbor candidates given a query q. We then refine the result with NN-expansion, i.e., we check the neighbors of q’s neighbors according to the approximate kNN graph to get closer neighbors. Algorithm 1 shows the detailed procedure.

```c++
Algorithm 1 EFANNA Search Algorithm
Input: data set D, query vector q, the number K of required nearest neighbors, EFANNA index (including tree set Stree and kNN graph G), the candidate pool size P, the expansion factor E, the iteration number I.
Output: approximate nearest neighbor set ANNS of the query
//E P I 都是参数
//通过tree+graph的方式，来搜索与query最接近的k个点
1: iter = 0
2: NodeList = NULL;
3: candidate set C = NULL;//候选节点set
4: suppose the maximal number of points of leaf node is Sleaf //最大叶子节点数
5: suppose the number of trees is Ntree //树的数量
6: then the maximal node check number is Nnode = P / Sleaf / Ntree + 1
    //需要检查的node最大node数，每棵树需要check的节点数（这里我理解，相比于叶子节点数，其他节点数比较少）
7: for all tree i in Stree do //先从所有的树中找到top node 符合搜索准则的。我理解是跟q有关的准则
8: 		Depth-first search i for top Nnode closest leaf nodes according to respective tree search criteria, add to NodeList
9: end for //对所有的树进行dfs，选出top Nnode的最近叶子节点，根据搜索规则
10: add the points belonging to the nodes in NodeList to C //把找到的points放入C
11: keep E points in C which are closest to q. //C中保留E points离q最近的
12: while iter < I do //我理解这里有点reverse knn的意思，邻居的邻居也很有可能与q接近的
13: 	candidate set CC = ;
14:		for all point n in C do //对于上面从树中找到的节点n
15: 		Sn is the neighbors of point n based on G.节点n在G的邻居集合是Sn
16: 		for all point nn in Sn do //对于Sn中的所有节点
17: 			if nn hasn’t been checked then //如果nn节点没有被check
18:					 put nn into CC.
19: 			end if
20: 		end for
21: 	end for
22: 	move all the points in CC to C and keep P points in C which are closest to q.
    	//还是要保持P个
23: 	iter = iter + 1
24: end while
25: return ANNS as the closet K points to q in C.
```

### 3.2 Tree building

先来两篇引用：hierarchical clustering [26] or randomized division tree [10]

作者使用的是randomized truncated KD-tree，跟其他randomized KD-tree的区别就是叶子节点有K个，K=10实验得出结论。不过作者说，这个树建立是不止在on-line search，还在knn graph建立阶段。

```c++
Algorithm 2 EFANNA Tree Building Algorithm
Input: the data set D, the number of trees T, the number of points in a leaf node K.
Output: the randomized truncated KD-tree set S
1:
2: function BUILDTREE(Node; PointSet)
3: 	if size of PointSet < K then //如果点集的大小比叶子节点数还少，就不建了
4: 		return
5: 	else
6:		Randomly choose dimension d. //KD-tree，随机选一个维度，作为树分裂的依据
7: 		Calculate the mean mid over PointSet on dimension d.
    	//不停的做分解，递归左树和左边节点，右树和右边节点集合。
8: 		Divide PointSet evenly into two subsets,LeftHalf and RightHalf, according to mid.
9: 		BUILDTREE(Node:LeftChild; LeftHalf)
10: 	BUILDTREE(Node:RightChild;RightHalf)
11: 	end if
12: return
13: end function
14:
15: for all i = 1 to T do //建立T棵树
16: 	BUILDTREE(Rooti;D) //从树根开始
17: 	Add Rooti to S. //树的集合
18: end for
```

### 3.3 knn graph construction

这个东西也分成两个阶段，第一阶段，考虑之前的tree构建，是有很多overlap的分割，所以做了**conquering step**，来get knn graph。第二阶段，NN-descent来调整knn graph。

Hierarchical Randomized divide-and-conquer

[12]是随机knn graph来初始化，然后用NN-descent来调整以期达到更高的acc。使用divide and conquer策略，对于appr graph construction来说，division part is easy，就是将数据集进行划分，就像树创建那样（参考3.2）。然后将sibling nodes从叶子节点开始进行merge，在一棵树上，我们需要conquer到root来得到一个联通的graph。但是直接从leaf到root，计算量会很大。参考论文[16]，可以多次随机划分（division）可以存在有overlap的数据集。这样的话，conquer就没有必要到root。另外，我们的更好地conquer策略是减少每层involved的节点以及使节点选择更有质量。(we make sure that we always choose the closest possible points for conquer，估计是在选节点的时候，采取了一些策略)。在下面的例子可以看到，如果我们知道q在node 8 并且还靠近node10，那么我们只需要考虑在node 10的点当conquer level1的4和5的时候（这里我理解：conquer从论文所述是指从叶子节点到root，节点q是叶子节点，也就是node8，因为node10与node8离得很近，在conquer4和5的时候，就需要考虑node10）。

换句话说，如果需要conquer 2个sibling non-leaf nodes（两个非叶子节点的兄弟节点）（叶子兄弟节点可以直接被conquer），对于一些在one subtree的点，我们只需要考虑最近的possible leaf node所在sibling subtree。因为其他叶子节点比较远，所以在他们的里面的点，也更可能远，因此，可以通过距离计算排除掉。（**大概理解论文说的是什么意思了，这里的8910数字代表的是点集，每个数字里面是一堆点，点在node8里面，离着最近的点应该在node10里面，通过层级模型，这些node可以看做是树模型的叶子节点，在构造层级模型的时候，从底向上conquer的时候，层级向上conquer的时候，并不需要所有点都参与，而是选择最近的点进行conquer，这样就减少了计算量**）。我们把tree看成一个分类模型，每一个叶子节点看成一类，可以看成把data space划分成矩形，如下面例子所示。area8和area10很近，但是是不同的area，通过node2来划分了他们。假设点q属于label 8。当要conquer level1的时候，就需要知道在subtree5（这个5划分了10和11）离q更近**（文章的意思是不需要考虑node3这个分支了，以及其他sub分支，如6和7等）**，从矩形图，q作为input to the classifier（subtree5），很显然，q会被分成label10，所以在合并到level1的时候，只需要考虑10里面的点。

![img](D:\work\note\searching\efanna_read.assets\7F639B5C-6286-48CC-9035-5167962A5F53.png)

所以，在处理每个level的时候，只需要考虑closest leaf node的点。

在论文中，使用的randomized KD-tree作为层级divide-and-conquer结构。

```c++
//conquer的目的是为了构造联通的图
Algorithm 3 Hierarchical Divide-and-Conquer Algorithm(kNN Graph Initialization)
Input: the data set D, the k in approximate kNN graph, the randomized truncated KD-tree set S built with Algorithm 2, the conquer-to depth Dep.
Output: approximate kNN graph G.
1: %%Division step //对数据集进行divide
2: Using Algorithm 2 to build tree, which leads to the input S //先用tree进行划分，建树
3: //每个叶子节点都是一个点，或数据集
4: %% Conquer step
5: G = Null;
6: for all point i in D do //对于所有数据集中的点i
7: 	Candidate pool C = ;
8: 	for all binary tree t in S do //对于所有的tree t
9: 		//在tree t里面搜索，与point i相关的leaf node（如何判断相关？）
		search in tree t with point i to the leaf node. 
10: 	add all the point in the leaf node to C. //将所有叶子节点加入到C里面
11: 	d = depth of the leaf node //当前遍历的这棵树的深度记下来
12: 	while d > Dep do //当深度大于conquer深度，Dep是个参数吧，也就是不遍历整棵树的深度
13: 		d = d - 1 //在保持深度的基础上，从叶子节点的上一级开始
14: 		Depth-first-search in the tree t with point i to depth d. Suppose N is the non-leaf node on the search path with depth d. Suppose Sib is the child node of N. And Sib is not on the search path of point i. //dfs tree t with point i到深度d，假设N是非叶子节点，Sib是N的孩子，并且Sib不是point i的search path
15: 		Depth-first-search to the leaf node in the subtree of Sib with point i . Add all the points in the leaf node to C.//dfs Sib，把所有的叶子节点都加到C里面
16: 	end while
17: end for
18: Reserve K closest points to i in C.
19:
20: Add C to G. //每个节点都应该有一个C
21: end for
//用上面的图来说明一下，如果point i 在节点4上，那么point i 的search path是1，2到4，不在search path上，就是1，2，5，节点5不在search path上面。所以合并的时候，要把节点5的叶子节点包括进去
```

graph refinement

这部分使用NN-descent来调整graph。主要思想还是要找better neighbors。更大的L和P能够保证更好的acc。

```c++
Algorithm 4 Approximate kNN Graph Refinement Algorithm
Input: an initial approximate k-nearest neighbor graph Ginit, data set D, maximum iteration number I, Candidate pool size P, new neighbor checking num L.
Output: an approximate kNN graph G.
1: iter = 0, G = Ginit
2: Graph Gnew records all the new added candidate neighbors of each point. Gnew = Ginit.
3: Graph Gold records all the old candidate neighbors of each point at previous iterations. Gold = Null;
4: Graph Grnew records all the new added reverse candidate neighbors of each point.
5: Graph Grold records all the old reverse candidate neighbors of each point.
    //这里的old只是指上一次迭代的结果（原来的结果），new是当次迭代的点（新发现的）
6: while iter < Imax do
7: 		Grnew = Null;, Grold = Null;. //new reverse candidate, old reverse candidate
8: 		for all point i in D do //所有dataset里面的点
9: 			NNnew is the neighbor set of point i in Gnew. //new added，初始值为G
10: 		NNold is the neighbor set of of point i in Gold. //old,初始值为空
    		//将i节点的所有邻居节点都遍历一遍
11: 		for all point j in NNnew do //节点i的所有邻居
12: 			for all point k in NNnew do //还是节点i的所有邻居
13: 				if j! = k then //如果是两个不同的邻居
14: 					calculate the distance between j and k. //计算两两之间距离
    					//把k添加到G里面（G一开始是随机的），并且k作为节点j的入节点，k标记为new
15: 					add k to j’s entry in G. mark k as new.//把k加到G里，作为j的入节点
    					//把j添加到G和Grnew里面，并且j作为k的入节点，Grnew是reverse候选节点
16: 					add j to k’s entry in G and Grnew. //把j加到G和Grnew里面，作为k入节点
    					//在这G里面，j和k互为邻居
17: 					//这里我理解就是当两个节点距离符合要求，那么就是互为最近邻节点
						mark j as new. //把j作为new
18: 				end if
19: 			end for
    			//遍历i节点在Gold里面的邻居，也是发现两两最近的节点，添加到图old中
20: 			for all point l in NNold do //上一轮节点i的邻居点集（初始为空）
21: 				calculate the distance between j and l. //计算old邻居和新邻居的距离
22: 				add l to j’s entry in G. mark l as old. //把l加到G，作为j的入节点，l为old
23: 				add j to l’s entry in G and Grold.//把j放到G和Grold
24: 				mark j as old. //把j标记为old
25: 			end for
26: 		end for
27: 	end for //将所有的点的邻居，都重新添加了一遍
    	//遍历所有D的点，保留P个最近的点，过滤
28: 	for all point i in D do 
29: 		Reserve the closest P points to i in respective entry of G.
30: 		entry of G.
31: 	end for
32: 	Gnew = Gold = Null;
		//遍历所有D中的点，遍历点的邻居，如果标记为new就放到Gnew，否则放到Gold
33: 	for all point i in D do
34: 		l = 0. NN is the neighbor set of i in G.
35: 		while l < L and l < P do
36: 			j = NN[l].
37: 			if j is marked as new then
38: 				add j to i’s entry in Gnew.
39: 				l = l + 1.
40: 			else
41: 				add j to i’s entry in Gold.
42: 			end if
43: 		end while
44: 	end for
45:		Gnew = Gnew union Grnew.
46: 	Gold = Gold union Grold
47: 	iter = iter + 1
48: end while
```

流程如上面所示，但是还有些不明白的，需要参考NN-descent文章了**[12]**

### 3.4 Online index updating

首先，当一个新的点来时，可以很轻松的插入，当要插入的点超过门限，只需要split的node。当树不平衡的时候（怎么判断是否平衡？平衡二叉树？？），就应该调整树的结构，这个调整也是很快的。第二，graph building可以接受stream data，也可以用同样的算法。

------

## 4 Experiments

使用的dataset是SIFT1M和GIST1M。程序C++，g++4.9 complier

评价指标：average recall

输入一个query point，return k points，需要从中检查true nearest neighbors。true nearest neighbors是R，通过算法返回的是R'，公式见论文。average recall是averaging over all the queries。

算法比较结论不写了。



补充一个英文基础知识：sibling指share同一个parent的节点；leaf nodes指没有children的节点；none leaf nodes指node不是leaf node。