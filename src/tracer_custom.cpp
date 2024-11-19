#include "tracer_custom.h"

#include "utils/exception.h"
#include "utils/vec_math.h"
#include "utils/gsrt_util.h"
#include "utils/Matrix.h"

using namespace util;

GaussiansKDTree::GaussiansKDTree() noexcept{}
GaussiansKDTree::~GaussiansKDTree() noexcept{}

GaussiansKDTree::GaussiansKDTree(GaussiansKDTree&& other) noexcept {}

void GaussiansKDTree::build(const GaussiansData& data){
    std::cout << "Building" << std::endl;
    using gsrt_util::icosahedron::n_verts;
    const float alpha_min = .01;
    aabbs.resize(data.numgs);
    for(int i = 0; i < data.numgs; ++i){
        const Matrix3x3 R = gsrt_util::construct_rotation(data.rotation[i]);
        float adaptive_scale = sqrt(2.*log(data.opacity[i]/alpha_min));
        float3 s = data.scaling[i]*adaptive_scale;
        for(int j = 0; j < n_verts; ++j){
            float3 v = gsrt_util::icosahedron::vertices[j];
            float3 w = R*(s*v)+data.xyz[i];
            aabbs[i].min = fminf(aabbs[i].min,w);
            aabbs[i].max = fmaxf(aabbs[i].max,w);
        }
    }
}

TracerCustom::TracerCustom(int8_t device){
}

TracerCustom::~TracerCustom() noexcept(true){
}

void TracerCustom::trace_rays(const TracingParams &tracing_params) {
   std::cout << "Tracing" << std::endl; 
}
