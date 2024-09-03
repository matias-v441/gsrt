struct Params {
    const float3* ray_origins;
    const float3* ray_directions;
    unsigned int* num_hits;
    unsigned int* hit_ids;
    float* hit_distances;
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
