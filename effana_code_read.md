代码结构

algorithm  -- **base_index**

​						**-- struct Point, struct Neighbor**

​						-- InsertIntoKnn

​						-- class InitIndex() (很多函数都是0)

​								-- void saveResults（）

​								-- void setSearchParams（）

​								-- void nnExpansion_kgraph() --待完成

​								-- void nnExpansion() --待完成

​								-- void update（）

​								-- void refineGraph（） --待完成

​					-- **hashing_index**

​								-- struct HASHINGIndexParams **继承params中的 struct IndexParams**

​								-- class HASHINGIndex **继承base_index 的class InitIndex()**

​											-- HASHINGIndex（）构造函数

​											-- void LoadCode32（）

​											-- void LoadCode64（）

​											-- void BuildHashTable32（）

​											-- void generateMask32（）

​											-- void BuildHashTable64（）

​											-- void generateMask64（）

​											-- void buildIndexImpl（）

​											-- void getNeighbors（），void getNeighbors32（），void getNeighbors64（）

​											-- void getNeighborsIEH32_nnexp（），void getNeighborsIEH32_kgraph（），void 													getNeighborsIEH64_nnexp（），void getNeighborsIEH64_kgraph（）

​											-- void outputVisitBucketNum()，void loadGraph(char* filename)

​					-- **init_indices**

​							-- inline InitIndex<DataType>* create_index_（）

​							-- create_index_by_type（）//选择创建kdtree索引还是hashing索引

​					-- **kdtreeub_index**

​							-- struct KDTreeUbIndexParams 继承params中的 struct IndexParams

​							-- class KDTreeUbIndex  继承base_index 的class InitIndex()

​									-- KDTreeUbIndex（）构造函数

​									-- struct Node

​									-- void loadIndex(),void saveIndex(),void loadTrees(),void saveTrees(),void loadGraph(),void saveGraph()

​									-- void SearchQueryToLeaf() 

​									-- void getSearchNodeList()

​									-- void getNeighbors()

​									-- void getNeighbors_nnexp(),void getNeighbors_kgraph()

​									-- int DepthFirstWrite()

​									-- struct Node* DepthFirstBuildTree()

​									-- void read_data(),void save_data()

​									**-- void meanSplit()** 已完成

​									**-- void planeSplit()** 已完成

​									**-- int selectDivision()** 已完成

​									**-- void getMergeLevelNodeList()** 已完成

​									**-- Node* SearchToLeaf()** 已完成

​									**-- void mergeSubGraphs()** 已完成

​									**-- static void GenRandom()**

​									**-- void DFSbuild()**,void DFStest()

​									**-- void buildTrees()** 已完成

​									**-- void initGraph(） 已完成**



general	  -- **distance**（与距离计算有关的加速函数）

​						--struct Candidate

​						--class Distance

​					--**matrix**

​						-- class Matrix

​					-- **params**

​						-- struct IndexParams

​						-- struct SearchParams

samples

​					-- efanna_index_buildall

​					-- efanna_index_buildgraph

​					-- efanna_index_buildtrees

​					-- efanna_search

​					-- evaluate

KDTree build

------

**kdtreeub_index**文件中，关于kd_tree和knn_graph部分的数据结构定义整理

```c++
  struct Point { //graph用
        unsigned id;//点编号
        float dist; //在graph中，代表该id与graph某点的距离，因为graph是以vector<heap>存储的
        bool flag;
        Point () {}
        Point (unsigned i, float d, bool f = true): id(i), dist(d), flag(f) {
        }//对变量进行初始化
        bool operator < (const Point &n) const{
            return this->dist < n.dist;
        }
    };
```

```c++
  struct Neighbor { //graph用
        std::shared_ptr<Lock> lock;
        float radius;
        float radiusM;
        Points pool; //侯选池
        unsigned L; //checking num L
        unsigned Range;
        bool found;
        std::vector<unsigned> nn_old;
        std::vector<unsigned> nn_new;
        std::vector<unsigned> rnn_old; //reverse nn
        std::vector<unsigned> rnn_new; //reverse nn
      Neighbor() : lock(std::make_shared<Lock>())
        {		}
        unsigned insert (unsigned id, float dist) {
            if (dist > radius) return pool.size();
            LockGuard guard(*lock);
            unsigned l = InsertIntoKnn(&pool[0], L, Point(id, dist, true));
            if (l <= L) {
                if (L + 1 < pool.size()) {
                    ++L;
                }
                else {
                    radius = pool[L-1].dist;
                }
            }
            return l;
        }
        template <typename C>
            void join (C callback) const {
                for (unsigned const i: nn_new) {
                    for (unsigned const j: nn_new) {
                        if (i < j) {
                            callback(i, j);
                        }
                    }
                    for (unsigned j: nn_old) {
                        callback(i, j);
                    }
                }
            }//end
      };
```

```c++
typedef std::set<Candidate<DataType>, std::greater<Candidate<DataType>> > CandidateHeap;
std::vector<CandidateHeap> knn_graph;  
std::vector<Neighbor>  nhoods;
```

```c++
	struct Node //tree节点
	{
		int DivDim; //分割的维度
		DataType DivVal;
		//明白了，指的是样本的范围，初始值应该是0 -- max_dim-1
		size_t StartIdx, EndIdx; //
		unsigned treeid;
		Node* Lchild, * Rchild;

		~Node() {
			if (Lchild!=NULL) Lchild->~Node();
			if (Rchild!=NULL) Rchild->~Node();
		}

	};
```

