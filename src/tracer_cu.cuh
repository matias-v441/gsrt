#pragma once
#include "gaussians_tracer.h"


namespace kdtree_impl{

	class GaussiansKDTree;
	struct Node;
	struct AABB { float3 min, max; };
	struct LeafData;

	class CUDA_Traversal {
	public:
		CUDA_Traversal(GaussiansKDTree&);
		~CUDA_Traversal();
		void rcast_kd_restart(const TracingParams&);
		void rcast_lin(const TracingParams&);
	private:
		
		Node* d_nodes = 0;
		AABB* d_aabbs = 0;
		int* d_leaves = 0;
		AABB* d_node_aabbs = 0;
		GaussiansData d_gaussians;
		AABB scene_vol;
		std::vector<int> leaves_data;
		GaussiansData gaussians;
		std::vector<Node> nodes;
		std::vector<AABB> node_aabbs;
	};
}