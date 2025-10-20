#pragma once
#include "utils/vec_math.h"
#include "../types.h"
#ifndef __CUDA_ARCH__
    #include <vector>
#endif

namespace gsrt::kd_tracer {

struct AABB { float3 min, max; };

class Node{
    int _axis;
    bool _isleaf;
    int dataid=-1;
    int right; 
    bool hasright;
    float p;
public:
    HOSTDEVICE INLINE int axis() const{
        return _axis;
    }
    HOSTDEVICE INLINE bool isleaf() const{
        return _isleaf;
    }
    HOSTDEVICE INLINE int data_id() const{
        return dataid;
    }
    HOSTDEVICE INLINE int right_id() const{
        return right;
    }
    HOSTDEVICE INLINE float cplane() const{
        return p;
    }
    INLINE void set_axis(int a){
        _axis = a; 
    }
    INLINE void set_right_id(int id){
        hasright = true;
        right = id;
    }
    INLINE void set_data_id(int id){
        _isleaf = true;
        dataid = id;
    }
    INLINE void set_cplane(float c){
        p = c;
    }
    HOSTDEVICE INLINE bool has_right() const{
        return hasright;
    }
    INLINE void set_leaf_empty(){
        _isleaf = true;
        dataid = -1;
    }
    HOSTDEVICE INLINE bool is_leaf_empty() const{
        return dataid == -1;
    }
};

// class SmallNode{
//     float p;
//     int data = __INT_MAX__; 
// public:
//     int axis() const{
//         return data & 3;
//     }
//     bool isleaf() const{
//         return data <= 0;
//     }
//     int data_id() const{
//         return -data;
//     }
//     int right_id() const{
//         return data >> 2;
//     }
//     float cplane() const{
//         return p;
//     }
//     void set_axis(int axis){
//         data += axis%3;
//     }
//     void set_right_id(int id){
//         data = (id<<2) + axis();
//     }
//     void set_data_id(int id){
//         data = -id;
//     }
//     void set_cplane(float c){
//         p = c;
//     }
//     bool has_right(){
//         return data == __INT_MAX__;
//     }
// };
// 


#ifndef __CUDA_ARCH__

struct LeafData
{
    std::vector<int> part_ids;
    std::vector<char> plane_masks;
};

struct ASData_Device;

struct ASData_Host{

    ASData_Host() = default;

    explicit ASData_Host(const ASData_Device&) noexcept(false);
    ~ASData_Host() noexcept(false);
    ASData_Host(const ASData_Host&) = delete;
    ASData_Host& operator=(const ASData_Host&) = delete;
    ASData_Host(ASData_Host&&);
    ASData_Host& operator=(ASData_Host&&);

    std::vector<AABB> aabbs;
    AABB scene_vol;
    std::vector<Node> nodes;
    std::vector<AABB> node_aabbs;
    std::vector<LeafData> leaves_data;
    std::vector<int> leaves_ids;
    GaussiansData data{};

    int max_depth = 0; 

    size_t get_size() const{
        return nodes.size() * sizeof(Node)
             + leaves_data.size()*sizeof(LeafData)
             + aabbs.size()*sizeof(AABB)
             + leaves_data.size()*sizeof(AABB);
    }

    void validate() const;
private:
    bool owns_gs_data = false;
};

#endif // __CUDA_ARCH__

struct ASData_Device {
    ASData_Device(int8_t device) : device(device) {}
#ifndef __CUDA_ARCH__
    explicit ASData_Device(const ASData_Host&) noexcept(false);
    ~ASData_Device() noexcept(false);
    ASData_Device(const ASData_Device&) = delete;
    ASData_Device& operator=(const ASData_Device&) = delete;
    ASData_Device(ASData_Device&&);
    ASData_Device& operator=(ASData_Device&&);
    void copy_from_host(const ASData_Host& kdtree) noexcept(false);
    void release() noexcept(false);
#endif

    size_t get_size() const{
        return 0;
    }

    struct DevData {
        Node* nodes = 0;
        AABB* aabbs = 0;
        int* leaves = 0;
        AABB* node_aabbs = 0;
        GaussiansData data{};
        AABB scene_vol;
    } _dd;

    int8_t device = -1;
private:
    bool owns_gs_data = false;
};

}