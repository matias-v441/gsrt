struct Params {
    const float3* ray_origins;
    const float3* ray_directions;

    int num_gs;
    const float3* gs_xyz;
    const float4* gs_rotation;
    const float3* gs_scaling;
    const float* gs_opacity;

    float3* radiance;
    float* transmittance;

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
