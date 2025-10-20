#include "structure.h"
#include <iostream>
#include <cassert>
#include <cstring>
#include "cuda_runtime.h"
#include "utils/exception_cuda.h"

using namespace gsrt::kd_tracer;

#ifndef __CUDA_ARCH__

void ASData_Host::validate() const{
    std::cout << "num. nodes: " << nodes.size() << std::endl;
    std::cout << "num. leaves: " << leaves_data.size() << std::endl;
    int max_leaf_size = 0;
    for(auto& leaf : leaves_data){
        max_leaf_size = max(leaf.part_ids.size(),max_leaf_size);
    }
    std::cout << "max leaf size: " << max_leaf_size << std::endl;
    std::cout << "max_depth: " << max_depth << std::endl;
    for(int i = 0; i < nodes.size(); ++i){
        if(nodes[i].isleaf()) continue;
        assert(nodes[i].has_right());
        int ax = nodes[i].axis();
        assert(nodes[i].cplane()==vec_c(node_aabbs[i+1].max,ax));
        assert(nodes[i].cplane()==vec_c(node_aabbs[nodes[i].right_id()].min,ax));
    }
    std::vector<int> containment_stats(data.numgs);
    for(int i = 0; i < nodes.size(); ++i){
        if(!nodes[i].isleaf() || nodes[i].is_leaf_empty()) continue;
        for(int part_id : leaves_data[nodes[i].data_id()].part_ids){
            containment_stats[part_id]++;
        }
    }
    int num_uncontained = 0;
    int num_overcontained = 0;
    int max_containment = 0;
    for(int part_id = 0; part_id < data.numgs; ++part_id){
        if(containment_stats[part_id] == 0){
            num_uncontained++;
        }
        if(containment_stats[part_id] > 2){
            num_overcontained++;
        }
        max_containment = std::max(max_containment, containment_stats[part_id]);
    }
    if(num_uncontained>0 || num_overcontained>0){
        std::cout << "WARNING: some gaussians not contained or overcontained!" << std::endl;
        std::cout << "num uncontained: " << num_uncontained << std::endl;
        std::cout << "num overcontained: " << num_overcontained << std::endl;
        std::cout << "max containment: " << max_containment << std::endl;
        std::cout << "total gaussians: " << data.numgs << std::endl;
    }
}

ASData_Host::ASData_Host(const ASData_Device& kdtree) noexcept(false){
    const auto& dev = kdtree._dd;
    if(dev.data.numgs != 0){
        data.numgs = dev.data.numgs;
        auto to_host = [](auto& dst, const auto dsrc, size_t count){
            using T = std::remove_pointer_t<std::remove_reference_t<std::remove_const_t<decltype(dst)>>>;
            dst = new T[count];
            CUDA_CHECK(cudaMemcpy(dst, reinterpret_cast<const void*>(dsrc), count * sizeof(T), cudaMemcpyDeviceToHost));
        };
        to_host(data.xyz, dev.data.xyz, dev.data.numgs);
        to_host(data.rotation, dev.data.rotation, dev.data.numgs);
        to_host(data.scaling, dev.data.scaling, dev.data.numgs);
        to_host(data.opacity, dev.data.opacity, dev.data.numgs);
        to_host(data.sh, dev.data.sh, dev.data.numgs*16);
        owns_gs_data = true;
    } 
}

ASData_Host::~ASData_Host() noexcept(false){
    if(owns_gs_data){
        delete[] data.xyz;
        delete[] data.rotation;
        delete[] data.scaling;
        delete[] data.opacity;
        delete[] data.sh;
    }
}

ASData_Host::ASData_Host(ASData_Host&& other){
    aabbs = std::move(other.aabbs);
    scene_vol = std::exchange(other.scene_vol, {});
    nodes = std::move(other.nodes);
    node_aabbs = std::move(other.node_aabbs);
    leaves_data = std::move(other.leaves_data);
    leaves_ids = std::move(other.leaves_ids);
    data = std::exchange(other.data, {});
    max_depth = std::exchange(other.max_depth, 0);
    owns_gs_data = std::exchange(other.owns_gs_data, false);
}

ASData_Host& ASData_Host::operator=(ASData_Host&& other){
    if (this != &other) {
        aabbs = std::move(other.aabbs);
        scene_vol = std::exchange(other.scene_vol, {});
        nodes = std::move(other.nodes);
        node_aabbs = std::move(other.node_aabbs);
        leaves_data = std::move(other.leaves_data);
        leaves_ids = std::move(other.leaves_ids);
        data = std::exchange(other.data, {});
        max_depth = std::exchange(other.max_depth, 0);
        owns_gs_data = std::exchange(other.owns_gs_data, false);
    }
    return *this;
}

void ASData_Device::copy_from_host(const ASData_Host& kdtree) noexcept(false){

    auto to_device = [](auto& dst, const auto* src, size_t count){
        using T = std::remove_const_t<std::remove_pointer_t<decltype(src)>>;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dst), count * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dst), src, count * sizeof(T), cudaMemcpyHostToDevice));
    };
    auto& data = _dd.data;
    if(data.numgs == 0){
        const auto& hdata = kdtree.data;
        data.numgs = hdata.numgs;
        to_device(data.xyz, hdata.xyz, hdata.numgs);
        to_device(data.rotation, hdata.rotation, hdata.numgs);
        to_device(data.scaling, hdata.scaling, hdata.numgs);
        to_device(data.opacity, hdata.opacity, hdata.numgs);
        to_device(data.sh, hdata.sh, hdata.numgs*16);
        owns_gs_data = true;
    }
    to_device(_dd.aabbs, kdtree.aabbs.data(), kdtree.aabbs.size());
    to_device(_dd.node_aabbs, kdtree.node_aabbs.data(), kdtree.node_aabbs.size()); // for debug drawing

    _dd.scene_vol = kdtree.scene_vol;

    std::vector<int> leaves_data;
    int offt = 0;
    std::vector<int> n_leaf_data_ids; 
    for(const auto& ld : kdtree.leaves_data){
        int start = offt;
        n_leaf_data_ids.push_back(start);
        int num_part = ld.part_ids.size();
        int masks_size = (num_part+3)>>2;
        offt += 1+num_part+masks_size;
        leaves_data.resize(offt);
        memcpy(leaves_data.data()+start,&num_part,sizeof(int));
        memcpy(leaves_data.data()+start+1,ld.part_ids.data(),num_part*sizeof(int));
        memcpy(leaves_data.data()+start+1+num_part,ld.plane_masks.data(),num_part);
    }
    assert(leaves_data.size()==offt);
    std::vector<Node> nodes(kdtree.nodes.begin(),kdtree.nodes.end());
    for(auto& node : nodes){
        if(node.isleaf()){
            node.set_data_id(n_leaf_data_ids[node.data_id()]);
        }
    }
    to_device(_dd.nodes, nodes.data(), nodes.size());
    to_device(_dd.leaves, leaves_data.data(), leaves_data.size());
}

void ASData_Device::release() noexcept(false){
    bool device_set = false;
    auto device_free = [dev=device,&device_set](auto& dptr){
        if (dptr != 0) {
            if (!device_set) { CUDA_CHECK(cudaSetDevice(dev)); device_set = true; }
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dptr)));
            dptr = 0;
        }
    };
    if(owns_gs_data){
        auto& data = _dd.data;
        device_free(data.xyz);
        device_free(data.rotation);
        device_free(data.scaling);
        device_free(data.opacity);
        device_free(data.sh);
        owns_gs_data = false;
    }
    device_free(_dd.aabbs);
    device_free(_dd.nodes);
    device_free(_dd.leaves);
    device_free(_dd.node_aabbs);
}

ASData_Device::ASData_Device(const ASData_Host& kdtree) noexcept(false) {
    copy_from_host(kdtree);
}

ASData_Device::~ASData_Device() noexcept(false) {
    if (this->device != -1) {
        release();
    }
    device = std::exchange(this->device, -1);
}

ASData_Device::ASData_Device(ASData_Device&& other){
    _dd.nodes = std::exchange(other._dd.nodes, nullptr);
    _dd.aabbs = std::exchange(other._dd.aabbs, nullptr);
    _dd.leaves = std::exchange(other._dd.leaves, nullptr);
    _dd.node_aabbs = std::exchange(other._dd.node_aabbs, nullptr);
    _dd.data = std::exchange(other._dd.data, {});
    _dd.scene_vol = std::exchange(other._dd.scene_vol, {});
    owns_gs_data = std::exchange(other.owns_gs_data, false);
    device = std::exchange(other.device, -1);
}

ASData_Device& ASData_Device::operator=(ASData_Device&& other) {
    if (this != &other) {
        _dd.nodes = std::exchange(other._dd.nodes, nullptr);
        _dd.aabbs = std::exchange(other._dd.aabbs, nullptr);
        _dd.leaves = std::exchange(other._dd.leaves, nullptr);
        _dd.node_aabbs = std::exchange(other._dd.node_aabbs, nullptr);
        _dd.data = std::exchange(other._dd.data, {});
        _dd.scene_vol = std::exchange(other._dd.scene_vol, {});
        owns_gs_data = std::exchange(other.owns_gs_data, false);
        device = std::exchange(other.device, -1);
    }
    return *this;
}

#endif // __CUDA_ARCH__