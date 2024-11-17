#include "tracer_custom.h"

#include "utils/exception.h"
#include "utils/vec_math.h"

//#define GLM_FORCE_SWIZZLE
//#include <glm/glm.hpp>

GaussiansKDTree::GaussiansKDTree() noexcept{}
GaussiansKDTree::~GaussiansKDTree() noexcept{}

GaussiansKDTree::GaussiansKDTree(GaussiansKDTree&& other) noexcept {}

// glm::mat3 construct_rotation(float4 vec){
//     glm::vec4 q = glm::normalize(glm::vec4(vec.x,vec.y,vec.z,vec.w));
//     glm::mat3 R(0.0f);
//     float r = q[0];
//     float x = q[1];
//     float y = q[2];
//     float z = q[3];
//     R[0][0] = 1. - 2. * (y*y + z*z);
//     R[1][0] = 2. * (x*y - r*z);
//     R[2][0] = 2. * (x*z + r*y);
//     R[0][1] = 2. * (x*y + r*z);
//     R[1][1] = 1. - 2. * (x*x + z*z);
//     R[2][1] = 2. * (y*z - r*x);
//     R[0][2] = 2. * (x*z - r*y);
//     R[1][2] = 2. * (y*z + r*x);
//     R[2][2] = 1. - 2. * (x*x + y*y);
//     return R;
// }

void GaussiansKDTree::build(const GaussiansData& data){
    std::cout << "Building" << std::endl;
    for(int i = 0; i < data.numgs; ++i){
        //construct_rotation();
    }
}

TracerCustom::TracerCustom(int8_t device){
}

TracerCustom::~TracerCustom() noexcept(true){
}

void TracerCustom::trace_rays(const TracingParams &tracing_params) {
   std::cout << "Tracing" << std::endl; 
}
