#include "tracer_custom.h"

#include "utils/exception.h"
#include "utils/vec_math.h"
#include "utils/gsrt_util.h"
#include "utils/Matrix.h"

#include <algorithm>

using namespace util;

void GaussiansKDTree::build(){
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

bool intersectAABB(const float3 min, const float3 max, const float3 orig, const float3 dir){
    float3 idir = 1.f/dir;
    float3 tc1 = (min-orig)*idir;
    float3 tc2 = (max-orig)*idir;
    float3 tc_min = fminf(tc1,tc2);
    float3 tc_max = fmaxf(tc1,tc2);
    float tmin = fmax(tc_min.x,fmax(tc_min.y,tc_min.z));
    float tmax = fmin(tc_max.x,fmin(tc_max.y,tc_max.z));
    return tmin < tmax;
}

void computeResponse(const GaussiansData& data, unsigned int gs_id,
                     const float3 o, const float3 d, float& resp, float& tmax){
    const float3 mu = data.xyz[gs_id];
    const float3 s = data.scaling[gs_id];
    Matrix3x3 R = gsrt_util::construct_rotation(data.rotation[gs_id]).transpose();
    constexpr float eps = 1e-6;
    R.setRow(0,R.getRow(0)/(s.x+eps));
    R.setRow(1,R.getRow(1)/(s.y+eps));
    R.setRow(2,R.getRow(2)/(s.z+eps));
    float3 og = R*(mu-o);
    float3 dg = R*d;
    tmax = dot(og,dg)/(dot(dg,dg)+eps);
    float3 samp = o+tmax*d;
    float3 x = R*(samp-mu);
    resp = data.opacity[gs_id]*exp(-dot(x,x));
}

void GaussiansKDTree::traverse(const TracingParams &tracing_params){
    std::cout << "Tracing" << std::endl; 
    std::cout << data.numgs << std::endl;
    for(int i = 0; i < tracing_params.num_rays; ++i){
        const float3 orig = tracing_params.ray_origins[i];
        const float3 dir = tracing_params.ray_directions[i];
        struct Hit{float resp, tmax;};
        std::vector<Hit> hits;
        for(int j = 0; j < data.numgs; ++j){
            if(intersectAABB(aabbs[j].min,aabbs[j].max,orig,dir)){
                float resp,tmax;
                computeResponse(data,j,orig,dir,resp,tmax);
                hits.push_back({resp,tmax});
            }
        }
        std::sort(hits.begin(),hits.end(),[](const Hit& a, const Hit& b){return a.tmax<b.tmax;});
        float3 rad_acc = make_float3(0.);
        float trans = 1.f;
        float3 color = make_float3(1.);
        for(const Hit& hit : hits){
            rad_acc += color*hit.resp*trans;
            trans *= (1-hit.resp);
        }
        tracing_params.radiance[i] = rad_acc;
    }
}
