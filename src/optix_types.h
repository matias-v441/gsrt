struct Params {
    const float3* ray_origins;
    const float3* ray_directions;
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
