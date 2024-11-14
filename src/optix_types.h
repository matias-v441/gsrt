struct Params {
    const float3* ray_origins;
    const float3* ray_directions;

    int num_gs;
    const float3* gs_xyz;
    const float4* gs_rotation;
    const float3* gs_scaling;
    const float* gs_opacity;
    const float3* gs_sh;

    const float3* gs_normals;
    
    int sh_deg;

    //int triagPerParticle = 20; // not supported
    //float resp_min = 0.01f;
    //const float3* triangles; 

    float3* radiance;
    float* transmittance;
    float3* debug_map_0;
    float3* debug_map_1;

    unsigned long long* num_its;
    unsigned long long* num_its_bwd;

    bool compute_grad;

    float3* grad_xyz;
    float4* grad_rotation;
    float3* grad_scale;
    float* grad_opacity;
    float3* grad_sh;

    float* grad_resp;

    bool* rad_clamped;
    float3* rad_sh;

    float3* dL_dC;

    OptixTraversableHandle handle;
};

struct RayGenData {
    // No data needed
};

struct MissData {
    // No data needed
};

struct HitGroupData {
    // No data needed
};
