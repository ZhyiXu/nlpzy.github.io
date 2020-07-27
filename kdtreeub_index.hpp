#ifndef EFANNA_KDTREE_UB_INDEX_H_
#define EFANNA_KDTREE_UB_INDEX_H_
#include "algorithm/base_index.hpp"
#include <fstream>
#include <ctime>
#include <string.h>
#include <random>
#include <queue>
//#include <bitset>
//using std::bitset;
#include <boost/dynamic_bitset.hpp>

namespace efanna{
struct KDTreeUbIndexParams : public IndexParams
{
	KDTreeUbIndexParams(bool rnn_used, int tree_num_total, int merge_level = 4, int epoches = 4, int check = 25, int myL = 30, int building_use_k = 10, int tree_num_build = 0, int myS = 10)
	{
		reverse_nn_used = rnn_used;
		init_index_type = KDTREE_UB;
		K = building_use_k;
		build_epoches = epoches;
		S = myS;
		ValueType treev;
		treev.int_val = tree_num_total;
		extra_params.insert(std::make_pair("trees",treev));
		ValueType treeb;
		treeb.int_val = tree_num_build > 0 ? tree_num_build : tree_num_total;
		extra_params.insert(std::make_pair("treesb",treeb));
		ValueType merge_levelv;
		merge_levelv.int_val = merge_level;
		extra_params.insert(std::make_pair("ml",merge_levelv));
		L = myL;
		Check_K = check;
	}
};
template <typename DataType>
class KDTreeUbIndex : public InitIndex<DataType>
{
public:

	typedef InitIndex<DataType> BaseClass;
	KDTreeUbIndex(const Matrix<DataType>& dataset, const Distance<DataType>* d, const IndexParams& params = KDTreeUbIndexParams(true,4)) :
		BaseClass(dataset,d,params)
	{
		std::cout<<"kdtree ub initial"<<std::endl;
		ExtraParamsMap::const_iterator it = params_.extra_params.find("trees");
		if(it != params_.extra_params.end()){
			TreeNum = (it->second).int_val;
#ifdef INFO
			std::cout << "Using kdtree to build "<< TreeNum << " trees in total" << std::endl;
#endif
		}
		else{
			TreeNum = 4;
#ifdef INFO
			std::cout << "Using kdtree to build "<< TreeNum << " trees in total" << std::endl;
#endif
		}
		SP.tree_num = TreeNum;

		it = params_.extra_params.find("treesb");
		if(it != params_.extra_params.end()){
			TreeNumBuild = (it->second).int_val;
#ifdef INFO
			std::cout << "Building kdtree graph with "<< TreeNumBuild <<" trees"<< std::endl;
#endif
		}
		else{
			TreeNumBuild = TreeNum;
#ifdef INFO
			std::cout << "Building kdtree graph with "<< TreeNumBuild <<" trees"<< std::endl;
#endif
		}

		it = params_.extra_params.find("ml");
		if(it != params_.extra_params.end()){
			ml = (it->second).int_val;
#ifdef INFO
			std::cout << "Building kdtree initial index with merge level "<< ml  << std::endl;
#endif
		}
		else{
			ml = -1;
#ifdef INFO
			std::cout << "Building kdtree initial index with max merge level "<< std::endl;
#endif
		}
		max_deepth = 0x0fffffff;
		error_flag = false;
	}

	void buildIndexImpl(){
#ifdef INFO
		clock_t s,f;
		s = clock();
#endif
		initGraph();

#ifdef INFO
		f = clock();
#endif

		std::cout << "initial graph finised"<< std::endl;
#ifdef INFO
		std::cout << "initial graph using time: "<< (f-s)*1.0/CLOCKS_PER_SEC<<" seconds"<< std::endl;
#endif

		if(error_flag){
			std::cout << "merge level deeper than tree, max merge deepth is" << max_deepth-1<<std::endl;
			return;
		}
		refineGraph();
	}
	struct Node
	{
		int DivDim; //分割的维度
		DataType DivVal;
		//指的是数据的index
		size_t StartIdx, EndIdx; //
		unsigned treeid;
		Node* Lchild, * Rchild;

		~Node() {
			if (Lchild!=NULL) Lchild->~Node();
			if (Rchild!=NULL) Rchild->~Node();
		}

	};

	void loadIndex(char* filename){
		read_data(filename);
	}
	void saveIndex(char* filename){

		size_t points_num = features_.get_rows();
		size_t feature_dim = features_.get_cols();
		save_data(filename, params_.K, points_num, feature_dim);
	}
	//algorithms copy and rewrite from flann
	void loadTrees(char* filename){
		std::ifstream in(filename, std::ios::binary|std::ios::in);
		if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-10087);}
		unsigned int K,tree_num;
		size_t dim,num;

		//read file head
		in.read((char*)&(K),sizeof(unsigned int));
		in.read((char*)&(tree_num),sizeof(unsigned int));
		in.read((char*)&(num),sizeof(size_t));
		in.read((char*)&(dim),sizeof(size_t));

		SP.tree_num = tree_num;

		//read trees

		tree_roots_.clear();
		for(unsigned int i=0;i<tree_num;i++){// for each tree
			int node_num, node_size;
			in.read((char*)&(node_num),sizeof(int));
			in.read((char*)&(node_size),sizeof(int));

			std::vector<struct Node *> tree_nodes;
			for(int j=0;j<node_num;j++){
				struct Node *tmp = new struct Node();
				in.read((char*)&(tmp->DivDim),sizeof(tmp->DivDim));
				in.read((char*)&(tmp->DivVal),sizeof(tmp->DivVal));
				in.read((char*)&(tmp->StartIdx),sizeof(tmp->StartIdx));
				in.read((char*)&(tmp->EndIdx),sizeof(tmp->EndIdx));
				in.read((char*)&(tmp->Lchild),sizeof(tmp->Lchild));
				in.read((char*)&(tmp->Rchild),sizeof(tmp->Rchild));
				tmp->Lchild = NULL;
				tmp->Rchild = NULL;
				tmp->treeid = i;
				tree_nodes.push_back(tmp);


			}
			//std::cout<<"build "<<i<<std::endl;
			struct Node *root = DepthFirstBuildTree(tree_nodes);
			if(root==NULL){ exit(-11); }
			tree_roots_.push_back(root);
		}

		//read index range
		LeafLists.clear();
		for(unsigned int i=0;i<tree_num;i++){

			std::vector<unsigned> leaves;
			for(unsigned int j=0;j<num; j++){
				unsigned leaf;
				in.read((char*)&(leaf),sizeof(int));
				leaves.push_back(leaf);
			}
			LeafLists.push_back(leaves);
		}
		in.close();
	}
	void saveTrees(char* filename){
		unsigned int K = params_.K;
		size_t num = features_.get_rows();
		size_t dim = features_.get_cols();
		std::fstream out(filename, std::ios::binary|std::ios::out);
		if(!out.is_open()){std::cout<<"open file error"<<std::endl;exit(-10086);}
		unsigned int tree_num = tree_roots_.size();

		//write file head
		out.write((char *)&(K), sizeof(unsigned int));
		out.write((char *)&(tree_num), sizeof(unsigned int));
		out.write((char *)&(num), sizeof(size_t)); //feature point number
		out.write((char *)&(dim), sizeof(size_t)); //feature dim

		//write trees
		typename std::vector<Node *>::iterator it;//int cnt=0;
		for(it=tree_roots_.begin(); it!=tree_roots_.end(); it++){
			//write tree nodes with depth first trace


			size_t offset_node_num = out.tellp();

			out.seekp(sizeof(int),std::ios::cur);

			unsigned int node_size = sizeof(struct Node);
			out.write((char *)&(node_size), sizeof(int));

			unsigned int node_num = DepthFirstWrite(out, *it);

			out.seekg(offset_node_num,std::ios::beg);

			out.write((char *)&(node_num), sizeof(int));

			out.seekp(0,std::ios::end);
			//std::cout<<"tree: "<<cnt++<<" written, node: "<<node_num<<" at offset " << offset_node_num <<std::endl;
		}

		if(LeafLists.size()!=tree_num){ std::cout << "leaf_size!=tree_num" << std::endl; exit(-6); }

		for(unsigned int i=0; i<tree_num; i++){
			for(unsigned int j=0;j<num;j++){
				out.write((char *)&(LeafLists[i][j]), sizeof(int));
			}
		}
		out.close();
	}
	void loadGraph(char* filename){
		std::ifstream in(filename,std::ios::binary);
		unsigned N;

		in.seekg(0,std::ios::end);
		std::ios::pos_type ss = in.tellg();
		size_t fsize = (size_t)ss;
		int dim;
		in.seekg(0,std::ios::beg);
		in.read((char*)&dim, sizeof(int));
		N = fsize / (dim+1) / 4;

		in.seekg(0,std::ios::beg);

		gs.resize(N);
		//M.resize(N);
		//norms.resize(N);
		for(unsigned i=0; i < N; i++){
			unsigned k;
			//DataType norm;
			in.read((char*)&k, sizeof(unsigned));
			//in.read((char*)&m, sizeof(unsigned));
			//in.read((char*)&norm, sizeof(DataType));
			//norms[i] = norm;
			//M[i] = m;
			gs[i].resize(k);

			for(unsigned j=0; j<k; j++){
				unsigned id;
				in.read((char*)&id, sizeof(unsigned));
				gs[i][j] = id;
			}
		}
		in.close();
	}
	/*
    void saveGraph(char* filename){
     std::ofstream out(filename,std::ios::binary);

     int dim = params_.K;//int meansize = 0;
     for(size_t i = 0; i < knn_graph.size(); i++){
       typename CandidateHeap::reverse_iterator it = knn_graph[i].rbegin();
       out.write((char*)&dim, sizeof(int));//meansize += knn_graph[i].size();
       for(size_t j =0; j < params_.K && it!= knn_graph[i].rend(); j++,it++ ){
         int id = it->row_id;
         out.write((char*)&id, sizeof(int));
       }
     }//meansize /= knn_graph.size();
     //std::cout << "size mean " << meansize << std::endl;
     out.close();
    }
	 */
	void saveGraph(char* filename){
		std::ofstream out(filename,std::ios::binary);
		unsigned N = gs.size();
		//out.write((char*)&N, sizeof(int));
		for(unsigned i=0; i < N; i++){
			unsigned k = gs[i].size();
			//unsigned m = M[i];
			//DataType norm = norms[i];
			out.write((char*)&k, sizeof(unsigned));
			//out.write((char*)&m, sizeof(unsigned));
			//out.write((char*)&norm, sizeof(DataType));
			for(unsigned j = 0; j < k; j++){
				unsigned id = gs[i][j];
				out.write((char*)&id, sizeof(unsigned));
			}
		}
		out.close();
	}
	//for nn search

	void SearchQueryToLeaf(Node* node, const DataType* q, unsigned dep, std::vector<Node*>& node_pool){
		if(node->Lchild != NULL && node->Rchild !=NULL){ //not leaf node
			if(q[node->DivDim] < node->DivVal){
				SearchQueryToLeaf(node->Lchild, q, dep, node_pool);
				if(node_pool.size() < dep)
					SearchQueryToLeaf(node->Rchild, q, dep, node_pool);
			}
			else{
				SearchQueryToLeaf(node->Rchild, q, dep, node_pool);
				if(node_pool.size() < dep)
					SearchQueryToLeaf(node->Lchild, q, dep, node_pool);
			}
		}
		else
			node_pool.push_back(node);
	}

	void getSearchNodeList(Node* node, const DataType* q, unsigned int lsize, std::vector<Node*>& vn){
		if(vn.size() >= lsize)
			return;

		if(node->Lchild != NULL && node->Rchild !=NULL){
			if(q[node->DivDim] < node->DivVal){
				getSearchNodeList(node->Lchild, q, lsize,  vn );
				getSearchNodeList(node->Rchild, q, lsize, vn);
			}else{
				getSearchNodeList(node->Rchild, q, lsize, vn);
				getSearchNodeList(node->Lchild, q, lsize, vn);
			}
		}else
			vn.push_back(node);
	}


	void getNeighbors(size_t searchK, const Matrix<DataType>& query){
		switch(SP.search_method){
		case 0:
			getNeighbors_nnexp(searchK, query);
			break;
		case 1:
			getNeighbors_kgraph(searchK, query);
			break;
		default:
			std::cout<<"no such searching method"<<std::endl;
		}

	}

	void getNeighbors_nnexp(size_t K, const Matrix<DataType>& query){
#ifdef INFO
		std::cout<<"using tree num "<< SP.tree_num<<std::endl;
#endif
		if(SP.tree_num > tree_roots_.size()){
			std::cout<<"wrong tree number"<<std::endl;return;
		}

		nn_results.clear();
		nn_results.resize(query.get_rows());
		unsigned dim = features_.get_cols();

		int resultSize = SP.extend_to;
		if (K > (unsigned)SP.extend_to)
			resultSize = K;


#pragma omp parallel for
		for(unsigned int cur = 0; cur < query.get_rows(); cur++){
			boost::dynamic_bitset<> tbflag(features_.get_rows(), false);
			boost::dynamic_bitset<> newflag(features_.get_rows(), true);
			tbflag.reset();
			newflag.set();

			std::vector<std::vector<Node*>> NodeCandi;
			NodeCandi.resize(SP.tree_num);

			const DataType* q_row = query.get_row(cur);
			_mm_prefetch((char *)q_row, _MM_HINT_T0);
			unsigned int lsize = SP.search_init_num*2 / (5*SP.tree_num) + 1;
			for(unsigned int i = 0; i < SP.tree_num; i++){
				getSearchNodeList(tree_roots_[i], q_row, lsize, NodeCandi[i]);
			}
			std::vector<int> pool(SP.search_init_num);
			unsigned int p = 0;
			for(unsigned int ni = 0; ni < lsize; ni++){
				for(unsigned int i = 0; i < NodeCandi.size(); i++){
					Node* leafn = NodeCandi[i][ni];
					for(size_t j = leafn->StartIdx; j < leafn->EndIdx && p < (unsigned int)SP.search_init_num; j++){
						size_t nn = LeafLists[i][j];
						if(tbflag.test(nn))continue;
						tbflag.set(nn);
						pool[p++]=(nn);
					}
					if(p >= (unsigned int)SP.search_init_num) break;
				}
				if(p >= (unsigned int)SP.search_init_num) break;
			}
			int base_n = features_.get_rows();
			while(p < (unsigned int)SP.search_init_num){
				unsigned int nn = rand() % base_n;
				if(tbflag.test(nn))continue;
				tbflag.set(nn);
				pool[p++]=(nn);
			}


			std::vector<std::pair<float,size_t>> result;
			//for(unsigned int i=0; i<pool.size();i++){
			//  _mm_prefetch((char *)features_.get_row(pool[i]), _MM_HINT_T0);
			//}
			unsigned cache_blocksz = 80;
			for(unsigned int i=0; i*cache_blocksz<pool.size();i++){
				unsigned s = i*cache_blocksz;
				unsigned t = s + cache_blocksz > pool.size() ? pool.size() : s+cache_blocksz;
				unsigned s_ = s;
				while(s<t){
					_mm_prefetch((char *)features_.get_row(pool[s]), _MM_HINT_T0);
					s++;
				}
				while(s_<t){
					result.push_back(std::make_pair(distance_->compare(q_row, features_.get_row(pool[s_]), dim),pool[s_]));
					s_++;
				}
			}
			std::partial_sort(result.begin(), result.begin() + resultSize, result.end());
			result.resize(resultSize);
			pool.clear();
			for(int j = 0; j < resultSize; j++)
				pool.push_back(result[j].second);

			int iter=0;
			std::vector<int> ids;
			while(iter++ < SP.search_epoches){
				ids.clear();
				for(unsigned j = 0; j < SP.extend_to ; j++){
					if(newflag.test( pool[j] )){
						newflag.reset(pool[j]);

						for(unsigned neighbor=0; neighbor < gs[pool[j]].size(); neighbor++){
							unsigned id = gs[pool[j]][neighbor];

							if(tbflag.test(id))continue;
							else tbflag.set(id);

							ids.push_back(id);
						}
					}
				}
				//for(unsigned int j=0; j<ids.size();j++){
				//_mm_prefetch((char *)features_.get_row(ids[j]), _MM_HINT_T0);
				//}
				for(size_t j = 0; j * cache_blocksz< ids.size(); j++){
					unsigned s = j * cache_blocksz;
					unsigned t = s + cache_blocksz > ids.size() ? ids.size() : s+cache_blocksz;
					unsigned s_ = s;
					while(s<t){
						_mm_prefetch((char *)features_.get_row(ids[s]), _MM_HINT_T0);
						s++;
					}
					while(s_<t){
						result.push_back(std::make_pair(distance_->compare(q_row, features_.get_row(ids[s_]), dim),ids[s_]));
						s_++;
					}
					//result.push_back(std::make_pair(distance_->compare(q_row, features_.get_row(ids[j]), dim),ids[j]));
				}
				std::partial_sort(result.begin(), result.begin() + resultSize, result.end());
				result.resize(resultSize);
				pool.clear();
				for(int j = 0; j < resultSize; j++)
					pool.push_back(result[j].second);
			}

			if(K<SP.extend_to)
				pool.resize(K);

			//nn_results.push_back(pool);
			std::vector<int>& res = nn_results[cur];
			for(unsigned i = 0; i < K ;i++)
				res.push_back(pool[i]);
		}
	}

	void getNeighbors_kgraph(size_t searchK, const Matrix<DataType>& query){
#ifdef INFO
		std::cout<<"using tree num "<< SP.tree_num<<std::endl;
#endif
		if(SP.tree_num > tree_roots_.size()){
			std::cout<<"wrong tree number"<<std::endl;return;
		}

		nn_results.clear();
		nn_results.resize(query.get_rows());
		unsigned dim = features_.get_cols();
		unsigned int lsize = SP.search_init_num*2 / (5*SP.tree_num) + 1;

		bool bSorted = true;
		unsigned pool_size = SP.search_epoches * SP.extend_to;
		if (pool_size >= (unsigned)SP.search_init_num){
			SP.search_init_num = pool_size;
			bSorted = false;
		}

#pragma omp parallel for
		for(unsigned int cur = 0; cur < query.get_rows(); cur++){
			std::mt19937 rng(1998);
			boost::dynamic_bitset<> flags(features_.get_rows(), false);

			std::vector<std::vector<Node*> > Vnl;
			Vnl.resize(SP.tree_num);
			const DataType* q_row = query.get_row(cur);
			_mm_prefetch((char *)q_row, _MM_HINT_T0);
			for(unsigned int i = 0; i < SP.tree_num; i++){
				getSearchNodeList(tree_roots_[i], q_row, lsize, Vnl[i]);
			}

			std::vector<int> pool(SP.search_init_num);
			unsigned int p = 0;
			for(unsigned int ni = 0; ni < lsize; ni++){
				for(unsigned int i = 0; i < Vnl.size(); i++){
					Node* leafn = Vnl[i][ni];
					for(size_t j = leafn->StartIdx; j < leafn->EndIdx && p < (unsigned int)SP.search_init_num; j++){
						size_t nn = LeafLists[i][j];
						if(flags.test(nn))continue;
						flags.set(nn);
						pool[p++]=(nn);
					}
					if(p >= (unsigned int)SP.search_init_num) break;
				}
				if(p >= (unsigned int)SP.search_init_num) break;
			}
			int base_n = features_.get_rows();
			while(p < (unsigned int)SP.search_init_num){
				unsigned int nn = rand() % base_n;
				if(flags.test(nn))continue;
				flags.set(nn);
				pool[p++]=(nn);
			}

			std::vector<std::pair<float,size_t>> result;
			unsigned cache_blocksz = 80;
			for(unsigned int i=0; i*cache_blocksz<pool.size();i++){
				unsigned s = i*cache_blocksz;
				unsigned t = s + cache_blocksz > pool.size() ? pool.size() : s+cache_blocksz;
				unsigned s_ = s;
				while(s<t){
					_mm_prefetch((char *)features_.get_row(pool[s]), _MM_HINT_T0);
					s++;
				}
				while(s_<t){
					result.push_back(std::make_pair(distance_->compare(q_row, features_.get_row(pool[s_]), dim),pool[s_]));
					s_++;
				}
			}
			if(bSorted){
				std::partial_sort(result.begin(), result.begin() + pool_size, result.end());
				result.resize(pool_size);
			}

			flags.reset();
			std::vector<Point> knn(searchK + SP.extend_to +1);
			std::vector<Point> results;
			for (unsigned iter = 0; iter < (unsigned)SP.search_epoches; ++iter) {

				unsigned L = 0;
				for(unsigned j=0; j < (unsigned)SP.extend_to ; j++){
					if(!flags.test(result[iter*SP.extend_to+j].second)){
						flags.set(result[iter*SP.extend_to+j].second);
						knn[L].id = result[iter*SP.extend_to+j].second;
						knn[L].dist = result[iter*SP.extend_to+j].first;
						knn[L].flag = true;
						L++;
					}
				}
				if(~bSorted){
					std::sort(knn.begin(), knn.begin() + L);
				}

				unsigned k =  0;
				while (k < L) {
					unsigned nk = L;
					if (knn[k].flag) {
						knn[k].flag = false;
						unsigned n = knn[k].id;

						//unsigned maxM = M[n];
						unsigned maxM = SP.extend_to;
						//if ((unsigned)SP.extend_to > maxM) maxM = SP.extend_to;
						auto const &neighbors = gs[n];
						if (maxM > neighbors.size()) {
							maxM = neighbors.size();
						}

						for(unsigned m = 0; m < maxM; ++m){
							_mm_prefetch((char *)features_.get_row(neighbors[m]), _MM_HINT_T0);
						}
						for (unsigned m = 0; m < maxM; ++m) {
							unsigned id = neighbors[m];
							//BOOST_VERIFY(id < graph.size());
							if (flags[id]) continue;
							flags[id] = true;

							DataType dist = distance_->compare(q_row, features_.get_row(id), dim);

							Point nn(id, dist);
							unsigned r = InsertIntoKnn(&knn[0], L, nn);
							//BOOST_VERIFY(r <= L);
							//if (r > L) continue;
							if (L + 1 < knn.size()) ++L;
							if (r < nk) {
								nk = r;
							}
						}
					}
					if (nk <= k) {
						k = nk;
					}
					else {
						++k;
					}
				}
				if (L > searchK) L = searchK;

				if (results.empty()) {
					results.reserve(searchK + 1);
					results.resize(L + 1);
					std::copy(knn.begin(), knn.begin() + L, results.begin());
				} else {
					for (unsigned int l = 0; l < L; ++l) {
						unsigned r = InsertIntoKnn(&results[0], results.size() - 1, knn[l]);
						if (r < results.size()  && results.size() < (searchK + 1)) {
							results.resize(results.size() + 1);
						}
					}
				}
			}

			std::vector<int>& res = nn_results[cur];
			for(size_t i = 0; i < searchK && i < results.size();i++)
				res.push_back(results[i].id);
		}
	}



	int DepthFirstWrite(std::fstream& out, struct Node *root){
		if(root==NULL) return 0;
		int left_cnt = DepthFirstWrite(out, root->Lchild);
		int right_cnt = DepthFirstWrite(out, root->Rchild);

		//std::cout << root->StartIdx <<":" << root->EndIdx<< std::endl;
		out.write((char *)&(root->DivDim), sizeof(root->DivDim));
		out.write((char *)&(root->DivVal), sizeof(root->DivVal));
		out.write((char *)&(root->StartIdx), sizeof(root->StartIdx));
		out.write((char *)&(root->EndIdx), sizeof(root->EndIdx));
		out.write((char *)&(root->Lchild), sizeof(root->Lchild));
		out.write((char *)&(root->Rchild), sizeof(root->Rchild));
		return (left_cnt + right_cnt + 1);
	}
	struct Node* DepthFirstBuildTree(std::vector<struct Node *>& tree_nodes){
		std::vector<Node*> root_serial;
		typename std::vector<struct Node*>::iterator it = tree_nodes.begin();
		for( ; it!=tree_nodes.end(); it++){
			Node* tmp = *it;
			size_t rsize = root_serial.size();
			if(rsize<2){
				root_serial.push_back(tmp);
				//continue;
			}
			else{
				Node *last1 = root_serial[rsize-1];
				Node *last2 = root_serial[rsize-2];
				if(last1->EndIdx == tmp->EndIdx && last2->StartIdx == tmp->StartIdx){
					tmp->Rchild = last1;
					tmp->Lchild = last2;
					root_serial.pop_back();
					root_serial.pop_back();
				}
				root_serial.push_back(tmp);
			}

		}
		if(root_serial.size()!=1){
			std::cout << "Error constructing trees" << std::endl;
			return NULL;
		}
		return root_serial[0];
	}
	void read_data(char *filename){
		std::ifstream in(filename, std::ios::binary|std::ios::in);
		if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-10087);}
		unsigned int K,tree_num;
		size_t dim,num;

		//read file head
		in.read((char*)&(K),sizeof(unsigned int));
		in.read((char*)&(tree_num),sizeof(unsigned int));
		in.read((char*)&(num),sizeof(size_t));
		in.read((char*)&(dim),sizeof(size_t));

		SP.tree_num = tree_num;

		//read trees

		tree_roots_.clear();
		for(unsigned int i=0;i<tree_num;i++){// for each tree
			int node_num, node_size;
			in.read((char*)&(node_num),sizeof(int));
			in.read((char*)&(node_size),sizeof(int));

			std::vector<struct Node *> tree_nodes;
			for(int j=0;j<node_num;j++){
				struct Node *tmp = new struct Node();
				in.read((char*)&(tmp->DivDim),sizeof(tmp->DivDim));
				in.read((char*)&(tmp->DivVal),sizeof(tmp->DivVal));
				in.read((char*)&(tmp->StartIdx),sizeof(tmp->StartIdx));
				in.read((char*)&(tmp->EndIdx),sizeof(tmp->EndIdx));
				in.read((char*)&(tmp->Lchild),sizeof(tmp->Lchild));
				in.read((char*)&(tmp->Rchild),sizeof(tmp->Rchild));
				tmp->Lchild = NULL;
				tmp->Rchild = NULL;
				tree_nodes.push_back(tmp);


			}
			//std::cout<<"build "<<i<<std::endl;
			struct Node *root = DepthFirstBuildTree(tree_nodes);
			if(root==NULL){ exit(-11); }
			tree_roots_.push_back(root);
		}

		//read index range
		LeafLists.clear();
		for(unsigned int i=0;i<tree_num;i++){

			std::vector<unsigned> leaves;
			for(unsigned int j=0;j<num; j++){
				unsigned leaf;
				in.read((char*)&(leaf),sizeof(int));
				leaves.push_back(leaf);
			}
			LeafLists.push_back(leaves);
		}

		//read knn graph
		knn_graph.clear();
		for(size_t i = 0; i < num; i++){
			CandidateHeap heap;
			for(size_t j =0; j < K ; j++ ){
				int id;
				in.read((char*)&id, sizeof(int));
				Candidate<DataType> can(id, -1);
				heap.insert(can);
			}
			knn_graph.push_back(heap);
		}
		in.close();
	}
	void save_data(char *filename, unsigned int K, size_t num, size_t dim){
		std::fstream out(filename, std::ios::binary|std::ios::out);
		if(!out.is_open()){std::cout<<"open file error"<<std::endl;exit(-10086);}
		unsigned int tree_num = tree_roots_.size();

		//write file head
		out.write((char *)&(K), sizeof(unsigned int));
		out.write((char *)&(tree_num), sizeof(unsigned int));
		out.write((char *)&(num), sizeof(size_t)); //feature point number
		out.write((char *)&(dim), sizeof(size_t)); //feature dim

		//write trees
		typename std::vector<Node *>::iterator it;//int cnt=0;
		for(it=tree_roots_.begin(); it!=tree_roots_.end(); it++){
			//write tree nodes with depth first trace


			size_t offset_node_num = out.tellp();

			out.seekp(sizeof(int),std::ios::cur);

			unsigned int node_size = sizeof(struct Node);
			out.write((char *)&(node_size), sizeof(int));

			unsigned int node_num = DepthFirstWrite(out, *it);

			out.seekg(offset_node_num,std::ios::beg);

			out.write((char *)&(node_num), sizeof(int));

			out.seekp(0,std::ios::end);
			//std::cout<<"tree: "<<cnt++<<" written, node: "<<node_num<<" at offset " << offset_node_num <<std::endl;
		}

		if(LeafLists.size()!=tree_num){ std::cout << "leaf_size!=tree_num" << std::endl; exit(-6); }

		for(unsigned int i=0; i<tree_num; i++){
			for(unsigned int j=0;j<num;j++){
				out.write((char *)&(LeafLists[i][j]), sizeof(int));
			}
		}

		//write knn-graph

		if(knn_graph.size()!=num){std::cout << "Error:" << std::endl; exit(-1);}
		for(size_t i = 0; i < knn_graph.size(); i++){
			typename CandidateHeap::reverse_iterator it = knn_graph[i].rbegin();
			for(size_t j =0; j < K && it!= knn_graph[i].rend(); j++,it++ ){
				int id = it->row_id;
				out.write((char*)&id, sizeof(int));
			}
		}

		out.close();
	}
	/*
    Node* divideTree(std::mt19937& rng, int* indices, size_t count, size_t offset){
      Node* node = new Node();
      if(count <= params_.TNS){
        node->DivDim = -1;
        node->Lchild = NULL;
        node->Rchild = NULL;
        node->StartIdx = offset;
        node->EndIdx = offset + count;
        //add points

        for(size_t i = 0; i < count; i++){
          for(size_t j = i+1; j < count; j++){
            DataType dist = distance_->compare(
                features_.get_row(indices[i]), features_.get_row(indices[j]), features_.get_cols());

            if(knn_graph[indices[i]].size() < params_.S || dist < knn_graph[indices[i]].begin()->distance){
              Candidate<DataType> c1(indices[j], dist);
              knn_graph[indices[i]].insert(c1);
              if(knn_graph[indices[i]].size() > params_.S)knn_graph[indices[i]].erase(knn_graph[indices[i]].begin());
            }
            else if(nhoods[indices[i]].nn_new.size() < params_.S * 2)nhoods[indices[i]].nn_new.push_back(indices[j]);
            if(knn_graph[indices[j]].size() < params_.S || dist < knn_graph[indices[j]].begin()->distance){
              Candidate<DataType> c2(indices[i], dist);
              knn_graph[indices[j]].insert(c2);
              if(knn_graph[indices[j]].size() > params_.S)knn_graph[indices[j]].erase(knn_graph[indices[j]].begin());
            }
            else if(nhoods[indices[j]].nn_new.size() < params_.S * 2)nhoods[indices[j]].nn_new.push_back(indices[i]);
          }
        }

      }else{
        int idx;
        int cutdim;
        DataType cutval;
        meanSplit(rng, indices, count, idx, cutdim, cutval);

        node->DivDim = cutdim;
        node->DivVal = cutval;
        node->StartIdx = offset;
        node->EndIdx = offset + count;
        node->Lchild = divideTree(rng, indices, idx, offset);
        node->Rchild = divideTree(rng, indices+idx, count-idx, offset+idx);
      }

      return node;
    }

    Node* divideTreeOnly(std::mt19937& rng, unsigned* indices, size_t count, size_t offset){
      Node* node = new Node();
      if(count <= params_.TNS){
        node->DivDim = -1;
        node->Lchild = NULL;
        node->Rchild = NULL;
        node->StartIdx = offset;
        node->EndIdx = offset + count;
        //add points

      }else{
        unsigned idx;
        unsigned cutdim;
        DataType cutval;
        meanSplit(rng, indices, count, idx, cutdim, cutval);

        node->DivDim = cutdim;
        node->DivVal = cutval;
        node->StartIdx = offset;
        node->EndIdx = offset + count;
        node->Lchild = divideTreeOnly(rng, indices, idx, offset);
        node->Rchild = divideTreeOnly(rng, indices+idx, count-idx, offset+idx);
      }

      return node;
    }
	 */
    //meanSplit(rng, &myids[0]+node->StartIdx, node->EndIdx - node->StartIdx, mid, cutdim, cutval); //在指定的维度，分裂树，对剩余的样本数
    //通过方差来划分树
	void meanSplit(std::mt19937& rng, unsigned* indices, unsigned count, unsigned& index, unsigned& cutdim, DataType& cutval){
		size_t veclen_ = features_.get_cols(); //feature的维度
		DataType* mean_ = new DataType[veclen_]; //每一个维度都有一个mean
		DataType* var_ = new DataType[veclen_];
		memset(mean_,0,veclen_*sizeof(DataType));
		memset(var_,0,veclen_*sizeof(DataType));

		/* Compute mean values.  Only the first SAMPLE_NUM values need to be
          sampled to get a good estimate.
		 */
		unsigned cnt = std::min((unsigned)SAMPLE_NUM+1, count);  //样本index范围取最小
		for (unsigned j = 0; j < cnt; ++j) { //样本数
			const DataType* v = features_.get_row(indices[j]); //某个样本
			for (size_t k=0; k<veclen_; ++k) {
				mean_[k] += v[k];
			}
		}
		DataType div_factor = DataType(1)/cnt;
		for (size_t k=0; k<veclen_; ++k) { //计算均值
			mean_[k] *= div_factor;
		}

		/* Compute variances (no need to divide by count). */

		for (unsigned j = 0; j < cnt; ++j) { //计算variance，这里没有再除以样本数做平均了，这里认为没有必要，少算一步
			const DataType* v = features_.get_row(indices[j]);
			for (size_t k=0; k<veclen_; ++k) {
				DataType dist = v[k] - mean_[k];
				var_[k] += dist * dist;
			}
		}

		/* Select one of the highest variance indices at random. */
		cutdim = selectDivision(rng, var_);

		cutval = mean_[cutdim]; //既然用var来决定切分的维度，那么切分的值就用mean来决定

		unsigned lim1, lim2;

		planeSplit(indices, count, cutdim, cutval, lim1, lim2);
		//index 对应输入的mid，用来划分数据节点的，将数据分成两个部分
		//这里我理解，为了保持树结构的平衡性，划分出来的数据集如果lim1，就是如果等于cutval的那个节点的位置索引，很偏左的话,就要调整到中间去，但是如果很偏右，好像就可以，还不做处理了，这里不是特别理解这样做的理由是什么
		//cut the subtree using the id which best balances the tree
		if (lim1>count/2) index = lim1;
		else if (lim2<count/2) index = lim2; //lim1<=count/2
		else index = count/2; //就是如果lim1<=count/2或者lim2>=count/2

		/* If either list is empty, it means that all remaining features
		 * are identical. Split in the middle to maintain a balanced tree.
		 */
		if ((lim1==count)||(lim2==0)) index = count/2;
		delete[] mean_;
		delete[] var_;
	}
    /*
     * 这一段有点傻逼，第一次看懵逼，因为对一个序列换了两次，要找到两个位置，一个是cutval之前的位置，一个是cutval之后第一个位置
     * 假设输入的数组是[7,8,9,5,4,6,1]，cut_val = 5,第一个for运行完之后结果是[1,4,9,5,8,6,7],lim1 = 1（4对应的位置）所以需要第二个for，运行完之后才是[1,4,5,9,8,6,7]，lim2 = 3（9对应的位置）
     */
	void planeSplit(unsigned* indices, unsigned count, unsigned cutdim, DataType cutval, unsigned& lim1, unsigned& lim2){
		/* Move vector indices for left subtree to front of list. */
		int left = 0;
		int right = count-1;
		//根据cutval和cutdim，来划分所有节点，用索引来划分，indices是node的标号
		for (;; ) {
			while (left<=right && features_.get_row(indices[left])[cutdim]<cutval) ++left; //左边的节点的cutdim的值应该小于cutval
			while (left<=right && features_.get_row(indices[right])[cutdim]>=cutval) --right; //右边的树应该大于cutval
			if (left>right) break; //直到相遇才退出循环
			//这里需要注意，虽然feature未做任何变动，但是由于索引位置改变，所以指向的元素也改变了，就相当于feature改变
			std::swap(indices[left], indices[right]); ++left; --right; //互换两个节点（输入样本）位置，从样本索引上来进行划分
		}
		lim1 = left;//lim1 is the id of the leftmost point <= cutval
		right = count-1;
		for (;; ) {
			while (left<=right && features_.get_row(indices[left])[cutdim]<=cutval) ++left;
			while (left<=right && features_.get_row(indices[right])[cutdim]>cutval) --right;
			if (left>right) break;
			std::swap(indices[left], indices[right]); ++left; --right;
		}
		lim2 = left;//lim2 is the id of the leftmost point >cutval
	}
	int selectDivision(std::mt19937& rng, DataType* v){ //有个问题，每一颗树应该都是不一样的，就是每次切分的点应该是随机的
		int num = 0;
		size_t topind[RAND_DIM]; //参数，设置为5的时候，效果最好
        /*
         * 下面这段操作的意思是，当有一个RAND_DIM之后，对输入的v，top_ind保留最大的RAND_DIM个数，并且还是按照降序排列
         * 选择num作为一个标志位，当num小于RAND_DIM的时候，往top_ind中放入维度索引，并且让top_ind中的元素按降序排列
         *                      当num等于RAND_DIM的时候，替换top_ind中最后的一个元素索引，让top_ind时刻保持元素都高于v中其他元素，num最终等于RAND_DIM
         * 随机生成一个rng，除以num求余数，实际是在【0，num)中选择一个随机数
         * 返回top中一个随机的特征维度进行划分
         * 因为KD_tree就是按照方差最大的原则来划分树的
         */
		//Create a list of the indices of the top RAND_DIM values.
		for (size_t i = 0; i < features_.get_cols(); ++i) { //遍历所有featuredim
			if ((num < RAND_DIM)||(v[i] > v[topind[num-1]])) { //如果num小于RAND_DIM的时候，或者方差的第i维大于某一维度的值，就记录下这个维度
				// Put this element at end of topind.
				if (num < RAND_DIM) {
					topind[num++] = i;            // Add to list.
				}
				else {
					topind[num-1] = i;         // Replace last element.
				}
				// Bubble end value down to right location by repeated swapping. sort the variance in decrease order
				//这里是进行一个互换操作，将输入的v内容按降序排序，最大的在前面
				int j = num - 1;
				while (j > 0  &&  v[topind[j]] > v[topind[j-1]]) {
					std::swap(topind[j], topind[j-1]);
					--j;
				}
			}
		}
		// Select a random integer in range [0,num-1], and return that index.
		int rnd = rng()%num; //每次选择的切分点都是随机的
		return (int)topind[rnd];
	}
	//getMergeLevelNodeList(tree_roots_[i], i ,0);
	//按层合并树，ml是merge level，结果装入mlNodeList里面，比较简单的递归
	void getMergeLevelNodeList(Node* node, size_t treeid, int deepth){
		if(node->Lchild != NULL && node->Rchild != NULL && deepth < ml){
			deepth++;
			getMergeLevelNodeList(node->Lchild, treeid, deepth);
			getMergeLevelNodeList(node->Rchild, treeid, deepth);
		}else if(deepth == ml){ //如果
			mlNodeList.push_back(std::make_pair(node,treeid));
		}else{
			error_flag = true;
			if(deepth < max_deepth)max_deepth = deepth;
		}
	}
	//根据根节点和start id 直到叶子节点，但是规则是沿着小于分裂值的方向去搜索
	Node* SearchToLeaf(Node* node, size_t id){
		if(node->Lchild != NULL && node->Rchild !=NULL){
			if(features_.get_row(id)[node->DivDim] < node->DivVal)
				return SearchToLeaf(node->Lchild, id);
			else
				return SearchToLeaf(node->Rchild, id);
		}
		else
			return node;
	}int cc = 0;
	//构造连通图，更新knn_graph和nhoods
	void mergeSubGraphs(size_t treeid, Node* node){
		if(node->Lchild != NULL && node->Rchild != NULL){ //还是对树进行操作
			mergeSubGraphs(treeid, node->Lchild);
			mergeSubGraphs(treeid, node->Rchild);

			size_t numL = node->Lchild->EndIdx - node->Lchild->StartIdx; //左边的树对应的节点
			size_t numR = node->Rchild->EndIdx - node->Rchild->StartIdx;
			size_t start,end;
			Node * root;
			//为什么处理节点少的子树？
			if(numL < numR){
				root = node->Rchild;
				start = node->Lchild->StartIdx;
                end = node->Lchild->EndIdx;
			}else{
				root = node->Lchild;
				start = node->Rchild->StartIdx;
				end = node->Rchild->EndIdx;
			}

			for(;start < end; start++){

				size_t feature_id = LeafLists[treeid][start]; //特征的编号

				Node* leaf = SearchToLeaf(root, feature_id); //直接到这颗树的叶子节点
				for(size_t i = leaf->StartIdx; i < leaf->EndIdx; i++){
					size_t tmpfea = LeafLists[treeid][i]; //也是编号
					//比较找到的叶子节点，与这棵树上的每个点的距离
					DataType dist = distance_->compare(
							features_.get_row(tmpfea), features_.get_row(feature_id), features_.get_cols());

					{LockGuard g(*nhoods[tmpfea].lock);
					//params_.S是
					//typedef std::set<Candidate<DataType>, std::greater<Candidate<DataType>> > CandidateHeap; 
					//std::vector<CandidateHeap> knn_graph;  std::vector<Neighbor>  nhoods; Neighbor结构体
					if(knn_graph[tmpfea].size() < params_.S || dist < knn_graph[tmpfea].begin()->distance){ //判断对应节点候选池的size，或者两个点的距离小于最大的距离（这里我理解begin就是里面最大的距离），塞进图
						Candidate<DataType> c1(feature_id, dist);
						knn_graph[tmpfea].insert(c1);
						if(knn_graph[tmpfea].size() > params_.S)knn_graph[tmpfea].erase(knn_graph[tmpfea].begin()); //去掉最大的那个


					} //这里是为了refinement？
					else if(nhoods[tmpfea].nn_new.size() < params_.S * 2){ //这个编号对应的nn_new如果小于blabla，就塞进去

						nhoods[tmpfea].nn_new.push_back(feature_id);

					}
					}
					{LockGuard g(*nhoods[feature_id].lock);
					//互为邻居
					if(knn_graph[feature_id].size() < params_.S || dist < knn_graph[feature_id].begin()->distance){
						Candidate<DataType> c1(tmpfea, dist);
						knn_graph[feature_id].insert(c1);
						if(knn_graph[feature_id].size() > params_.S)knn_graph[feature_id].erase(knn_graph[feature_id].begin());

					}
					else if(nhoods[feature_id].nn_new.size() < params_.S * 2){

						nhoods[feature_id].nn_new.push_back(tmpfea);

					}
					}
				}
			}
		}
	}

	typedef std::set<Candidate<DataType>, std::greater<Candidate<DataType>> > CandidateHeap;


protected:
	enum
	{
		/**
		 * To improve efficiency, only SAMPLE_NUM random values are used to
		 * compute the mean and variance at each level when building a tree.
		 * A value of 100 seems to perform as well as using all values.
		 */
		SAMPLE_NUM = 100,
		/**
		 * Top random dimensions to consider
		 *
		 * When creating random trees, the dimension on which to subdivide is
		 * selected at random from among the top RAND_DIM dimensions with the
		 * highest variance.  A value of 5 works well.
		 */
		RAND_DIM=5
	};

	int TreeNum;
	int TreeNumBuild;
	int ml;   //merge_level
	int max_deepth;
	int veclen_;
	//DataType* var_;
	omp_lock_t rootlock;
	bool error_flag;
	//DataType* mean_;
	std::vector<Node*> tree_roots_;
	std::vector< std::pair<Node*,size_t> > mlNodeList;
	std::vector<std::vector<unsigned>> LeafLists;
	USING_BASECLASS_SYMBOLS

	//kgraph code

	static void GenRandom (std::mt19937& rng, unsigned *addr, unsigned size, unsigned N) {
		for (unsigned i = 0; i < size; ++i) {
			addr[i] = rng() % (N - size);
		}
		std::sort(addr, addr + size);
		for (unsigned i = 1; i < size; ++i) {
			if (addr[i] <= addr[i-1]) {
				addr[i] = addr[i-1] + 1;
			}
		}
		unsigned off = rng() % N;
		for (unsigned i = 0; i < size; ++i) {
			addr[i] = (addr[i] + off) % N;
		}
	}

    //DFSbuild(node, rng, &myids[0]+node->StartIdx, node->EndIdx-node->StartIdx, node->StartIdx);
	void DFSbuild(Node* node, std::mt19937& rng, unsigned* indices, unsigned count, unsigned offset){
		//omp_set_lock(&rootlock);
		//std::cout<<node->treeid<<":"<<offset<<":"<<count<<std::endl;
		//omp_unset_lock(&rootlock);
		//如果剩余的样本数，小于tree node size，就不再划分了，最下层的根节点数，也就是说，最下层根节点至少得有10个点，只要大于10，就继续拆分
		if(count <= params_.TNS){ //tree node size
			node->DivDim = -1;
			node->Lchild = NULL;
			node->Rchild = NULL;
			node->StartIdx = offset;
			node->EndIdx = offset + count;
			//add points

		}else{//剩余节点数大于10，就需要继续划分树
			unsigned idx;
			unsigned cutdim;
			DataType cutval;
			meanSplit(rng, indices, count, idx, cutdim, cutval);
			node->DivDim = cutdim;
			node->DivVal = cutval;
			node->StartIdx = offset;
			node->EndIdx = offset + count;
			Node* nodeL = new Node(); Node* nodeR = new Node();
			node->Lchild = nodeL;
			nodeL->treeid = node->treeid;
			DFSbuild(nodeL, rng, indices, idx, offset);
			node->Rchild = nodeR;
			nodeR->treeid = node->treeid;
			DFSbuild(nodeR, rng, indices+idx, count-idx, offset+idx);
		}
	}

	void DFStest(unsigned level, unsigned dim, Node* node){
		if(node->Lchild !=NULL){
			DFStest(++level, node->DivDim, node->Lchild);
			//if(level > 15)
			std::cout<<"dim: "<<node->DivDim<<"--cutval: "<<node->DivVal<<"--S: "<<node->StartIdx<<"--E: "<<node->EndIdx<<" TREE: "<<node->treeid<<std::endl;
			if(node->Lchild->Lchild ==NULL){
				std::vector<unsigned>& tmp = LeafLists[node->treeid];
				for(unsigned i = node->Rchild->StartIdx; i < node->Rchild->EndIdx; i++)
					std::cout<<features_.get_row(tmp[i])[node->DivDim]<<" ";
				std::cout<<std::endl;
			}
		}
		else if(node->Rchild !=NULL){
			DFStest(++level, node->DivDim, node->Rchild);
		}
		else{
			std::cout<<"dim: "<<dim<<std::endl;
			std::vector<unsigned>& tmp = LeafLists[node->treeid];
			for(unsigned i = node->StartIdx; i < node->EndIdx; i++)
				std::cout<<features_.get_row(tmp[i])[dim]<<" ";
			std::cout<<std::endl;
		}
	}
	void buildTrees(){
		unsigned N = features_.get_rows(); //样本数量
		unsigned seed = 1998;
		std::mt19937 rng(seed);
		nhoods.resize(N); //std::vector<Neighbor>  nhoods; //把邻居的样本容量也是搞成N
		g.resize(N); //std::vector<std::vector<Point> > g; 点集
		boost::dynamic_bitset<> visited(N, false);
		knn_graph.resize(N); //std::vector<CandidateHeap> knn_graph; 每一个节点都有一个heap，knn是由若干heap组成，节点数与样本数相同
		for (auto &nhood: nhoods) {
			//nhood.nn_new.resize(params_.S * 2);
			nhood.pool.resize(params_.L+1); //对应论文中侯选池pool，L为侯选池长度
			nhood.radius = std::numeric_limits<float>::max();
		}

		//build tree
		std::vector<int> indices(N);
		LeafLists.resize(TreeNum); //有几棵树对应几个LeafList
		std::vector<Node*> ActiveSet;
		std::vector<Node*> NewSet;
		for(unsigned i = 0; i < (unsigned)TreeNum; i++){ //对于每一棵树，初始化根节点
			Node* node = new Node;
			node->DivDim = -1;
			node->Lchild = NULL;
			node->Rchild = NULL;
			node->StartIdx = 0; //样本数目N，这里对应的是分裂过程中，要划分的样本数是不断减少的，每一层都有一个对应的开始和截止节点的index
			node->EndIdx = N;
			node->treeid = i;
			tree_roots_.push_back(node);
			ActiveSet.push_back(node);
		}
#pragma omp parallel for
		for(unsigned i = 0; i < N; i++)indices[i] = i; //每个样本都有一个标号，后面是通过标号来找样本的
#pragma omp parallel for
		for(unsigned i = 0; i < (unsigned)TreeNum; i++){ //对于每一棵树
			std::vector<unsigned>& myids = LeafLists[i]; //取每个leaflist的地址
			myids.resize(N);
			std::copy(indices.begin(), indices.end(),myids.begin()); //将样本的索引复制给myids，然后shuffle //std::copy(start, end, container.begin());
			std::random_shuffle(myids.begin(), myids.end());
		}
		omp_init_lock(&rootlock);
		while(!ActiveSet.empty() && ActiveSet.size() < 1100){ //Activeset按层装节点，不为空的时候循环
#pragma omp parallel for //循环并行化
			for(unsigned i = 0; i < ActiveSet.size(); i++){ //处理每棵树，通过循环，一层一层的分裂
				Node* node = ActiveSet[i]; //从root开始
				unsigned mid; //中值，用来划分数据集
				unsigned cutdim; //分裂的维度是哪一维
				DataType cutval;
				std::mt19937 rng(seed ^ omp_get_thread_num());
				std::vector<unsigned>& myids = LeafLists[node->treeid]; //随机选择一棵树
				//也就是说，每一次分裂对应的样本都是不一样的，所以每棵树都不一样
				//myids进入的时候对应的内容是全部的样本乱序，分解的时候，从start index开始，到end index截止，因为不是所有样本都会分裂的，具体原因是下面
				meanSplit(rng, &myids[0]+node->StartIdx, node->EndIdx - node->StartIdx, mid, cutdim, cutval); //在指定的维度，分裂树，node->EndIdx - node->StartIdx 待分裂的样本数量
                //以当前节点为根节点时：
				node->DivDim = cutdim; //分裂在第几维
				node->DivVal = cutval; //分裂的值
				//node->StartIdx = offset;
				//node->EndIdx = offset + count;
				/*
				 * 虽然这里startIdx和EndIdx，都是顺序的，但是其指向feature内部的元素并不是顺序了，这里startIdx和EndIdx是索引myids的，myids是meansplit的输入indice
				 * 但meansplit内部一直在操作换为indice
				 * 加入原来indice数组是按顺序的，indice = [0,1,2,3,4],也分别对应feature的0,1,2,3,4行（样本）
				 * 经过meansplit后，indice = 【3,1,0,2,4】，indice通过start和end按顺序索引，但是其实对应的样本id就不一样了
				 */
				Node* nodeL = new Node(); Node* nodeR = new Node(); //左边的节点，右边的节点
				nodeR->treeid = nodeL->treeid = node->treeid; //定义左右孩子的treeid
				nodeL->StartIdx = node->StartIdx;//左子树对应的要分裂的点开始index
				nodeL->EndIdx = node->StartIdx+mid; //通过meanSplit找到了分裂的点的index就是mid了，这时候，把数据根据mid分成前后两部分，然后继续分裂,这个分裂，分裂是有overlap的
				nodeR->StartIdx = nodeL->EndIdx;
				nodeR->EndIdx = node->EndIdx;
				node->Lchild = nodeL; //左右孩子进行连接
				node->Rchild = nodeR;
				omp_set_lock(&rootlock);
				if(mid>params_.S)NewSet.push_back(nodeL);
				if(nodeR->EndIdx - nodeR->StartIdx > params_.S)NewSet.push_back(nodeR);
				omp_unset_lock(&rootlock);
			}
			ActiveSet.resize(NewSet.size()); //剩余的孩子节点也会被放到activeSet里面
			std::copy(NewSet.begin(), NewSet.end(),ActiveSet.begin());
			NewSet.clear();
		}
		//下面真的没看懂了，因为ActiveSet已经empty了，怎么还能循环，或者ActiveSet大于1100的时候，进入这个循环？1100是怎么来的
#pragma omp parallel for
		for(unsigned i = 0; i < ActiveSet.size(); i++){ //应该是ActiveSet已经超额了，上述的循环退出了，这里为什么要分成两个循环来写？
			Node* node = ActiveSet[i];
			//omp_set_lock(&rootlock);
			//std::cout<<i<<":"<<node->EndIdx-node->StartIdx<<std::endl;
			//omp_unset_lock(&rootlock);
			std::mt19937 rng(seed ^ omp_get_thread_num());
			std::vector<unsigned>& myids = LeafLists[node->treeid]; //通过activeset里面的树根，把对应的leaf节点全找出来
			DFSbuild(node, rng, &myids[0]+node->StartIdx, node->EndIdx-node->StartIdx, node->StartIdx);
		}
	}
    void outputVisitBucketNum(){}

	void initGraph(){
		//initial
		unsigned N = features_.get_rows();
		unsigned seed = 1998;
		std::mt19937 rng(seed);
		nhoods.resize(N);
		g.resize(N);
		boost::dynamic_bitset<> visited(N, false);
		knn_graph.resize(N);
		for (auto &nhood: nhoods) {
			//nhood.nn_new.resize(params_.S * 2);
			nhood.pool.resize(params_.L+1);
			nhood.radius = std::numeric_limits<float>::max();
		}
		//注释参考buildTrees（），是一样的，也是先建多个树,没有核对有什么差异，因为作者又写了一遍
		//build tree
		std::vector<int> indices(N);
		LeafLists.resize(TreeNum);
		std::vector<Node*> ActiveSet;
		std::vector<Node*> NewSet;
		for(unsigned i = 0; i < (unsigned)TreeNum; i++){
			Node* node = new Node;
			node->DivDim = -1;
			node->Lchild = NULL;
			node->Rchild = NULL;
			node->StartIdx = 0;
			node->EndIdx = N;
			node->treeid = i;
			tree_roots_.push_back(node);
			ActiveSet.push_back(node);
		}
#pragma omp parallel for
		for(unsigned i = 0; i < N; i++)indices[i] = i;
#pragma omp parallel for
		for(unsigned i = 0; i < (unsigned)TreeNum; i++){
			std::vector<unsigned>& myids = LeafLists[i];
			myids.resize(N);
			std::copy(indices.begin(), indices.end(),myids.begin());
			std::random_shuffle(myids.begin(), myids.end());
		}
		omp_init_lock(&rootlock);
		while(!ActiveSet.empty() && ActiveSet.size() < 1100){
#pragma omp parallel for
			for(unsigned i = 0; i < ActiveSet.size(); i++){
				Node* node = ActiveSet[i];
				unsigned mid;
				unsigned cutdim;
				DataType cutval;
				std::mt19937 rng(seed ^ omp_get_thread_num());
				std::vector<unsigned>& myids = LeafLists[node->treeid];

				meanSplit(rng, &myids[0]+node->StartIdx, node->EndIdx - node->StartIdx, mid, cutdim, cutval);

				node->DivDim = cutdim;
				node->DivVal = cutval;
				//node->StartIdx = offset;
				//node->EndIdx = offset + count;
				Node* nodeL = new Node(); Node* nodeR = new Node();
				nodeR->treeid = nodeL->treeid = node->treeid;
				nodeL->StartIdx = node->StartIdx;
				nodeL->EndIdx = node->StartIdx+mid;
				nodeR->StartIdx = nodeL->EndIdx;
				nodeR->EndIdx = node->EndIdx;
				node->Lchild = nodeL;
				node->Rchild = nodeR;
				omp_set_lock(&rootlock);
				if(mid>params_.S)NewSet.push_back(nodeL); //参数nn sets max size
				if(nodeR->EndIdx - nodeR->StartIdx > params_.S)NewSet.push_back(nodeR);
				omp_unset_lock(&rootlock);
			}
			ActiveSet.resize(NewSet.size());
			std::copy(NewSet.begin(), NewSet.end(),ActiveSet.begin());
			NewSet.clear();
		}
#pragma omp parallel for
		for(unsigned i = 0; i < ActiveSet.size(); i++){
			Node* node = ActiveSet[i];
			//omp_set_lock(&rootlock);
			//std::cout<<i<<":"<<node->EndIdx-node->StartIdx<<std::endl;
			//omp_unset_lock(&rootlock);
			std::mt19937 rng(seed ^ omp_get_thread_num());
			std::vector<unsigned>& myids = LeafLists[node->treeid];
			DFSbuild(node, rng, &myids[0]+node->StartIdx, node->EndIdx-node->StartIdx, node->StartIdx);
		}
		//DFStest(0,0,tree_roots_[0]);
		//build tree completed
        //遍历每棵树，每棵树先merge成node list，然后再把node list merge 起来，按层就行merge
        //结果装入mlNodeList
		for(size_t i = 0; i < (unsigned)TreeNumBuild; i++){
			getMergeLevelNodeList(tree_roots_[i], i ,0);
		}

		//构造连通图，输入mlNodeList更新knn_graph和nhoods
#pragma omp parallel for	
		for(size_t i = 0; i < mlNodeList.size(); i++){
			mergeSubGraphs(mlNodeList[i].second, mlNodeList[i].first);
		}


#pragma omp parallel
		{
#ifdef _OPENMP
			std::mt19937 rng(seed ^ omp_get_thread_num());
#else
			std::mt19937 rng(seed);
#endif
			std::vector<unsigned> random(params_.S + 1);
			//我理解这里后面是对nhoods进行整理，因为涉及到graph refinement，更新nhoods.nn_new
#pragma omp for
			for (unsigned n = 0; n < N; ++n) {//所有样本节点
				auto &nhood = nhoods[n];
				Points &pool = nhood.pool; //pool size侯选池
				if(nhood.nn_new.size()<params_.S*2){
					nhood.nn_new.resize(params_.S*2);
					GenRandom(rng, &nhood.nn_new[0], nhood.nn_new.size(), N);
				}


				GenRandom(rng, &random[0], random.size(), N);
				nhood.L = params_.S;
				nhood.Range = params_.S; //范围？
				while(knn_graph[n].size() < params_.S){
					unsigned rand_id = rng() % N; //还再继续随机找节点
					DataType dist = distance_->compare(
							features_.get_row(n), features_.get_row(rand_id), features_.get_cols());
					Candidate<DataType> c(rand_id,dist);
					knn_graph[n].insert(c);
				}

				//omp_set_lock(&rootlock);
				//if(knn_graph[n].size() < nhood.L)std::cout<<n<<":"<<knn_graph[n].size()<<std::endl;
				//omp_unset_lock(&rootlock);
				unsigned i = 0;
				typename CandidateHeap::reverse_iterator it = knn_graph[n].rbegin(); //rbegin，逆向迭代器，指向最后一个
				for (unsigned l = 0; l < nhood.L; ++l) {
					if (random[i] == n) ++i;
					auto &nn = nhood.pool[l];
					nn.id = it->row_id;//random[i++];
					nhood.nn_new[l] = it->row_id;
					nn.dist = it->distance;//distance_->compare(features_.get_row(n), features_.get_row(nn.id), features_.get_cols());
					nn.flag = true;it++;
					//if(it == knn_graph[n].rend())break;
				}
				sort(pool.begin(), pool.begin() + nhood.L);//侯选池点排序
			}
		}
		knn_graph.clear();
#ifdef INFO
		std::cout<<"initial completed"<<std::endl;
#endif
	}

};

}
#endif
