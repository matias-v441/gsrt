#include "gaussians_tracer.h"

#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <cassert>
#include <iostream>
#include <iomanip>
#include <cstring>

#include "optix_types.h"
#include "utils/exception.h"
#include "utils/vec_math.h"

#define GLM_FORCE_SWIZZLE
#include <glm/glm.hpp>

// These structs represent the data blocks of our SBT records
template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

static void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
              << message << "\n";
}

GaussiansTracer::GaussiansTracer(int8_t device)
    : device(device){

    // Initialize fields
    context = nullptr;

    // Switch to active device
    CUDA_CHECK(cudaSetDevice(device));

    // Load PTX first to make sure it exists

    char log[2048];  // For error reporting from OptiX creation functions

    //
    // Initialize CUDA and create OptiX context
    //
    {
        // Initialize CUDA
        // Warning: CUDA should have been already initialized at this point!!
        CUDA_CHECK(cudaFree(0));

        // Initialize the OptiX API, loading all API entry points
        OPTIX_CHECK(optixInit());

        // Specify context options
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &context_log_cb;
        options.logCallbackLevel = 4;

        // Associate a CUDA context (and therefore a specific GPU) with this
        // device context
        CUcontext cuCtx = 0;  // zero means take the current context
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
    }

    trace_rays_pipeline = std::move(TraceRaysPipeline(context, device));
    gaussians_structure = std::move(GaussiansAS(context, device));
}

GaussiansTracer::GaussiansTracer(GaussiansTracer &&other)
    : context(std::exchange(other.context, nullptr)),
        device(std::exchange(other.device, -1)),
        gaussians_structure(std::move(other.gaussians_structure)),
        trace_rays_pipeline(std::move(other.trace_rays_pipeline)) {}

GaussiansTracer::~GaussiansTracer() noexcept(false) {
    // We call the destructor manually here to ensure correct destruction order
    gaussians_structure.~GaussiansAS();
    trace_rays_pipeline.~TraceRaysPipeline();

    if (context != nullptr && device != -1) {
        CUDA_CHECK(cudaSetDevice(device));
        OPTIX_CHECK(optixDeviceContextDestroy(std::exchange(context, nullptr)));
    }
}


TraceRaysPipeline::TraceRaysPipeline(const OptixDeviceContext &context, int8_t device)
     : device(device), context(context)
    {
    // Initialize fields
    OptixPipelineCompileOptions pipeline_compile_options = {};

    // Switch to active device
    CUDA_CHECK(cudaSetDevice(device));

    // Load PTX first to make sure it exists

    char log[2048];  // For error reporting from OptiX creation functions

    //
    // Create module
    //
    std::vector<OptixPayloadType> payloadTypes;
    {
        unsigned int payload_flags =
             OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ | OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_WRITE 
            | OPTIX_PAYLOAD_SEMANTICS_AH_READ  | OPTIX_PAYLOAD_SEMANTICS_AH_WRITE
            //| OPTIX_PAYLOAD_SEMANTICS_MS_READ  | OPTIX_PAYLOAD_SEMANTICS_MS_WRITE
            //| OPTIX_PAYLOAD_SEMANTICS_CH_READ  | OPTIX_PAYLOAD_SEMANTICS_CH_WRITE
            ;
        constexpr int payload_size_fwd = 8;
        unsigned int semantics_fwd[payload_size_fwd] = {
                                    payload_flags,
                                    payload_flags,
                                    payload_flags,
                                    payload_flags,

                                    payload_flags,
                                    payload_flags,

                                    payload_flags,

                                    payload_flags,
        };
        constexpr int payload_size_bwd = 11;
        unsigned int semantics_bwd[payload_size_bwd] = {
                                    payload_flags,
                                    payload_flags,
                                    payload_flags,
                                    payload_flags,

                                    payload_flags,
                                    payload_flags,

                                    payload_flags,

                                    payload_flags,
                                    payload_flags,
                                    payload_flags,
                                    payload_flags,
        };

        OptixPayloadType payloadTypeFwd;
        payloadTypeFwd.payloadSemantics = semantics_fwd;
        payloadTypeFwd.numPayloadValues = payload_size_fwd;
        payloadTypes.push_back(payloadTypeFwd);

        OptixPayloadType payloadTypeBwd;
        payloadTypeBwd.payloadSemantics = semantics_bwd;
        payloadTypeBwd.numPayloadValues = payload_size_bwd;
        payloadTypes.push_back(payloadTypeBwd);

        OptixModuleCompileOptions module_compile_options = {};
        module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_compile_options.numPayloadTypes = payloadTypes.size();
        module_compile_options.payloadTypes = payloadTypes.data();

// The following is not supported in Optix 7.2
// #define XSTR(x) STR(x)
// #define STR(x) #x
// #pragma message "Optix ABI version is: " XSTR(OPTIX_ABI_VERSION)
#if (OPTIX_ABI_VERSION > 54)
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#else
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#endif

        pipeline_compile_options.usesMotionBlur = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_compile_options.numPayloadValues = 0;
        // https://forums.developer.nvidia.com/t/how-to-calculate-numattributevalues-of-optixpipelinecompileoptions/110833
        pipeline_compile_options.numAttributeValues = 2;
#ifdef DEBUG  // Enables debug exceptions during optix launches. This may incur significant performance cost and should only be done during development.
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
        pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

        std::string input = TraceRaysPipeline::load_ptx_data();
        size_t sizeof_log = sizeof(log);

        
#if OPTIX_VERSION < 70700
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(
            context,
            &module_compile_options,
            &pipeline_compile_options,
            input.c_str(),
            input.size(),
            log,
            &sizeof_log,
            &module));
#else
    
        OPTIX_CHECK_LOG(optixModuleCreate(
            context,
            &module_compile_options,
            &pipeline_compile_options,
            input.c_str(),
            input.size(),
            log,
            &sizeof_log,
            &module));
#endif
    }

    //
    // Create program groups
    //
    {
        OptixProgramGroupOptions empty_program_group_options = {};

        OptixProgramGroupDesc raygen_prog_group_desc = {};
        raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module = module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &raygen_prog_group_desc,
            1,  // num program groups
            &empty_program_group_options,
            log,
            &sizeof_log,
            &raygen_prog_group));

        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module = nullptr;//module;
        miss_prog_group_desc.miss.entryFunctionName = nullptr;//"__miss__ms";
        //sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &miss_prog_group_desc,
            1,  // num program groups
            &empty_program_group_options,
            //log,
            //&sizeof_log,
            nullptr,
            nullptr,
            &miss_prog_group));

        OptixProgramGroupOptions hg_fwd_program_group_options = {};
        hg_fwd_program_group_options.payloadType = &payloadTypes[0];
        OptixProgramGroupDesc hitgroup_prog_group_desc = {};
        hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prog_group_desc.hitgroup.moduleCH = nullptr;
        hitgroup_prog_group_desc.hitgroup.moduleAH = module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__fwd";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &hitgroup_prog_group_desc,
            1,  // num program groups
            &hg_fwd_program_group_options,
            log,
            &sizeof_log,
            &hitgroup_prog_group));

        OptixProgramGroupOptions hg_bwd_program_group_options = {};
        hg_bwd_program_group_options.payloadType = &payloadTypes[1];
        OptixProgramGroupDesc bwd_hitgroup_prog_group_desc = {};
        bwd_hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        bwd_hitgroup_prog_group_desc.hitgroup.moduleCH = nullptr;
        bwd_hitgroup_prog_group_desc.hitgroup.moduleAH = module;
        bwd_hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
        bwd_hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__bwd";
        sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &bwd_hitgroup_prog_group_desc,
            1,  // num program groups
            &hg_bwd_program_group_options,
            log,
            &sizeof_log,
            &bwd_hitgroup_prog_group));
    }

    //
    // Link pipeline
    //
    {
        const uint32_t max_trace_depth = 1;
        OptixProgramGroup program_groups[] = {
            raygen_prog_group,
            miss_prog_group,
            hitgroup_prog_group,
            bwd_hitgroup_prog_group
            };

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth = max_trace_depth;
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK_LOG(optixPipelineCreate(
            context,
            &pipeline_compile_options,
            &pipeline_link_options,
            program_groups,
            sizeof(program_groups) / sizeof(program_groups[0]),
            log,
            &sizeof_log,
            &pipeline));

        OptixStackSizes stack_sizes = {};
        for (auto &prog_group : program_groups) {
            OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline));
        }

        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;
        OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                                               0,  // maxCCDepth
                                               0,  // maxDCDEpth
                                               &direct_callable_stack_size_from_traversal,
                                               &direct_callable_stack_size_from_state, &continuation_stack_size));
        OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                                              direct_callable_stack_size_from_state, continuation_stack_size,
                                              1  // maxTraversableDepth
                                              ));
    }

    //
    // Set up shader binding table
    //
    {
        CUdeviceptr raygen_record;
        const size_t raygen_record_size = sizeof(RayGenSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raygen_record), raygen_record_size));
        RayGenSbtRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(raygen_record),
            &rg_sbt,
            raygen_record_size,
            cudaMemcpyHostToDevice));

        CUdeviceptr miss_record;
        size_t miss_record_size = sizeof(MissSbtRecord);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&miss_record), miss_record_size));
        MissSbtRecord ms_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(miss_record),
            &ms_sbt,
            miss_record_size,
            cudaMemcpyHostToDevice));

        std::vector<HitGroupSbtRecord> hg_records;
        HitGroupSbtRecord hg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
        hg_records.push_back(hg_sbt);
        HitGroupSbtRecord bwd_hg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(bwd_hitgroup_prog_group, &bwd_hg_sbt));
        hg_records.push_back(bwd_hg_sbt);

        CUdeviceptr hitgroup_records_base;
        size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
        CUDA_CHECK(cudaMalloc(
            reinterpret_cast<void **>(&hitgroup_records_base),
            hitgroup_record_size*hg_records.size()));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(hitgroup_records_base),
            hg_records.data(),
            hitgroup_record_size*hg_records.size(),
            cudaMemcpyHostToDevice));


        sbt.raygenRecord = raygen_record;
        sbt.missRecordBase = miss_record;
        sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = hitgroup_records_base;
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
        sbt.hitgroupRecordCount = hg_records.size();
    }

    {
        CUDA_CHECK(cudaStreamCreate(&stream));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_param), sizeof(Params)));
    }
}


TraceRaysPipeline::TraceRaysPipeline(TraceRaysPipeline &&other) noexcept
    : device(std::exchange(other.device, -1)),
      context(std::exchange(other.context, nullptr)),
      pipeline(std::exchange(other.pipeline, nullptr)),
      raygen_prog_group(std::exchange(other.raygen_prog_group, nullptr)),
      miss_prog_group(std::exchange(other.miss_prog_group, nullptr)),
      hitgroup_prog_group(std::exchange(other.hitgroup_prog_group, nullptr)),
      bwd_hitgroup_prog_group(std::exchange(other.bwd_hitgroup_prog_group, nullptr)),
      module(std::exchange(other.module, nullptr)),
      sbt(std::exchange(other.sbt, {})),
      stream(std::exchange(other.stream, nullptr)),
      d_param(std::exchange(other.d_param, 0)) {}
      


TraceRaysPipeline::~TraceRaysPipeline() noexcept(false) {
    const auto device = std::exchange(this->device, -1);
    if (device == -1) {
        return;
    }
    CUDA_CHECK(cudaSetDevice(device));
    if (d_param != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(d_param, 0))));
    if (sbt.raygenRecord != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.raygenRecord, 0))));
    if (sbt.missRecordBase != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.missRecordBase, 0))));
    if (sbt.hitgroupRecordBase != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.hitgroupRecordBase, 0))));
    if (sbt.callablesRecordBase)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.callablesRecordBase, 0))));
    if (sbt.exceptionRecord)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.exceptionRecord, 0))));
    sbt = {};
    if (stream != nullptr)
        CUDA_CHECK(cudaStreamDestroy(std::exchange(stream, nullptr)));
    if (pipeline != nullptr)
        OPTIX_CHECK(optixPipelineDestroy(std::exchange(pipeline, nullptr)));
    if (raygen_prog_group != nullptr)
        OPTIX_CHECK(optixProgramGroupDestroy(std::exchange(raygen_prog_group, nullptr)));
    if (miss_prog_group != nullptr)
        OPTIX_CHECK(optixProgramGroupDestroy(std::exchange(miss_prog_group, nullptr)));
    if (hitgroup_prog_group != nullptr)
        OPTIX_CHECK(optixProgramGroupDestroy(std::exchange(hitgroup_prog_group, nullptr)));
    if (bwd_hitgroup_prog_group != nullptr)
        OPTIX_CHECK(optixProgramGroupDestroy(std::exchange(bwd_hitgroup_prog_group, nullptr)));
    if (module != nullptr)
        OPTIX_CHECK(optixModuleDestroy(std::exchange(module, nullptr)));
}

void TraceRaysPipeline::trace_rays(const GaussiansAS *gaussians_structure,
                                   const TracingParams &tracing_params
                                   ) {
    CUDA_CHECK(cudaSetDevice(device));

    {
        Params params;
        params.handle = gaussians_structure->gas_handle();

        params.ray_origins = tracing_params.ray_origins;
        params.ray_directions = tracing_params.ray_directions;

        auto& gs = gaussians_structure->device_gaussians();
        params.num_gs = gs.numgs;
        params.gs_xyz = gs.xyz;
        params.gs_rotation = gs.rotation;
        params.gs_scaling = gs.scaling;
        params.gs_opacity = gs.opacity;
        params.gs_sh = gs.sh;
        params.sh_deg = gs.sh_deg;
        params.gs_normals = gs.normals;

        params.radiance = tracing_params.radiance;
        params.transmittance = tracing_params.transmittance;
        params.debug_map_0 = tracing_params.debug_map_0;
        params.debug_map_1 = tracing_params.debug_map_1;
        params.num_its = tracing_params.num_its;
        params.num_its_bwd = tracing_params.num_its_bwd;

        params.compute_grad = tracing_params.compute_grad;
        params.grad_xyz = tracing_params.grad_xyz;
        params.grad_rotation = tracing_params.grad_rotation;
        params.grad_scale = tracing_params.grad_scale;
        params.grad_opacity = tracing_params.grad_opacity;
        params.grad_sh = tracing_params.grad_sh;

        params.dL_dC = tracing_params.dL_dC;

        auto& sbs = gaussians_structure->device_scene_buffers();
        params.rad_clamped = sbs.rad_clamped;
        params.rad_sh = sbs.rad_sh;

        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(d_param),
            &params, sizeof(params),
            cudaMemcpyHostToDevice));
        OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(params), &sbt,
                                tracing_params.width, tracing_params.height, 1));
        //OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(params), &sbt,
        //                        tracing_params.num_rays, 1, 1));
        CUDA_SYNC_CHECK();
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
}


GaussiansAS::GaussiansAS() noexcept
    : device(-1),
      context(nullptr),
      gas_handle_(0),
      d_gas_output_buffer(0),
      d_vertices(0),
      d_triangles(0) {}


GaussiansAS::GaussiansAS(GaussiansAS &&other) noexcept
    : device(std::exchange(other.device, -1)),
      context(std::exchange(other.context, nullptr)),
      gas_handle_(std::exchange(other.gas_handle_, 0)),
      d_gas_output_buffer(std::exchange(other.d_gas_output_buffer, 0)),
      d_vertices(std::exchange(other.d_vertices, 0)),
      d_triangles(std::exchange(other.d_triangles, 0)),
      d_gaussians(std::exchange(other.d_gaussians, {})),
      d_scene_buffers(std::exchange(other.d_scene_buffers, {})) {}

void GaussiansAS::release() {
    bool device_set = false;
    auto device_free = [dev=device,&device_set](auto& dptr){
        if (dptr != 0) {
            if (!device_set) { CUDA_CHECK(cudaSetDevice(dev)); device_set = true; }
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dptr)));
            dptr = 0;
        }
    };
    device_free(d_gas_output_buffer);
    gas_handle_ = 0;
    device_free(d_vertices);
    device_free(d_triangles);
    device_free(d_gaussians.xyz);
    device_free(d_gaussians.rotation);
    device_free(d_gaussians.scaling);
    device_free(d_gaussians.opacity);
    device_free(d_gaussians.sh);
    device_free(d_gaussians.normals);
    device_free(d_scene_buffers.rad_clamped);
    device_free(d_scene_buffers.rad_sh);
}

GaussiansAS::~GaussiansAS() noexcept(false) {
    if (this->device != -1) {
        release();
    }
    const auto device = std::exchange(this->device, -1);
}

namespace icosahedron{
    constexpr float x = sqrt(.4f*(5.f+sqrt(5.f))) *.5f;
    constexpr float y = x*(1.f+sqrt(5.f))*.5f;
    constexpr int n_verts = 12;
    constexpr int n_faces = 20;
    float3 vertices[n_verts] = {
        make_float3(-x,y,0.f),make_float3(x,y,0.f),make_float3(-x,-y,0.f),make_float3(x,-y,0.f),
        make_float3(0.f,-x,y),make_float3(0.f,x,y),make_float3(0.f,-x,-y),make_float3(0.f,x,-y),
        make_float3(y,0.f,-x),make_float3(y,0.f,x),make_float3(-y,0.f,-x),make_float3(-y,0.f,x),
        };
    uint3 triangles[n_faces] = {
        make_uint3(0,11,5), make_uint3(0,5,1), make_uint3(0,1,7), make_uint3(0,7,10), make_uint3(0,10,11),
        make_uint3(1,5,9), make_uint3(5,11,4), make_uint3(11,10,2), make_uint3(10,7,6), make_uint3(7,1,8),
        make_uint3(3,9,4), make_uint3(3,4,2), make_uint3(3,2,6), make_uint3(3,6,8), make_uint3(3,8,9),
        make_uint3(4,9,5), make_uint3(2,4,11), make_uint3(6,2,10), make_uint3(8,6,7), make_uint3(9,8,1),
    };
}


namespace triangle{

    constexpr int n_verts = 3;
    constexpr int n_faces = 1;
    float3 vertices[n_verts] = {make_float3(-1.,-1.,0.),make_float3(-1.,1.,0.),make_float3(1.,0.,0.)};
    uint3 triangles[n_faces] = {make_uint3(2,1,0)};
}


namespace tetrahedron{

    constexpr int n_verts = 4;
    constexpr int n_faces = 4;
    float3 vertices[n_verts] = {make_float3(-1.,-1.,0.),make_float3(-1.,1.,0.),make_float3(1.,0.,0.),make_float3(0.,0.,1.)};
    uint3 triangles[n_faces] = {make_uint3(2,1,0), make_uint3(2,1,3), make_uint3(0,1,3), make_uint3(2,0,3)};
    void construct(){
        for(int i = 0; i < n_verts; ++i){
            vertices[i] -= make_float3(-.25f,0.f,.25f);
        }
    }
}

glm::mat3 construct_rotation(float4 vec){
    glm::vec4 q = glm::normalize(glm::vec4(vec.x,vec.y,vec.z,vec.w));
    glm::mat3 R(0.0f);
    float r = q[0];
    float x = q[1];
    float y = q[2];
    float z = q[3];
    R[0][0] = 1. - 2. * (y*y + z*z);
    R[1][0] = 2. * (x*y - r*z);
    R[2][0] = 2. * (x*z + r*y);
    R[0][1] = 2. * (x*y + r*z);
    R[1][1] = 1. - 2. * (x*x + z*z);
    R[2][1] = 2. * (y*z - r*x);
    R[0][2] = 2. * (x*z - r*y);
    R[1][2] = 2. * (y*z + r*x);
    R[2][2] = 1. - 2. * (x*x + y*y);
    return R;
}

void construct_primitives(const GaussiansData& data, std::vector<float3>& vertices,
     std::vector<uint3>& triangles, std::vector<float3>& normals){
    namespace primitive = icosahedron;
    //primitive::construct();
    using primitive::n_verts;
    using primitive::n_faces;
    vertices.resize(n_verts*data.numgs);
    triangles.resize(n_faces*data.numgs);
    normals.resize(n_faces*data.numgs);
    const float alpha_min = .01;
    for(int i = 0; i < data.numgs; ++i){

        glm::mat3 R = construct_rotation(data.rotation[i]);
        float adaptive_scale = sqrt(2.*log(data.opacity[i]/alpha_min));
        float3 s = data.scaling[i]*adaptive_scale;
        glm::vec3 scale = glm::vec3(s.x,s.y,s.z);

        for(int j = 0; j < n_verts; ++j){
            float3 v = primitive::vertices[j];
            glm::vec3 w = R*(scale*glm::vec3(v.x,v.y,v.z));
            vertices[j+i*n_verts] = make_float3(w.x,w.y,w.z)+data.xyz[i];
        }

        for(int j = 0; j < n_faces; ++j){
            uint3 triag = primitive::triangles[j];
            triag += make_uint3(i*n_verts);
            triangles[j+i*n_faces] = triag;
            const float3 &v1 = vertices[triag.x];
            const float3 &v2 = vertices[triag.y];
            const float3 &v3 = vertices[triag.z];
            normals[j+i*n_faces] = normalize(cross(v2-v1,v3-v1));
        }
    }
}

void GaussiansAS::build(const GaussiansData& data) {
    release();

    CUDA_CHECK(cudaSetDevice(device));

    std::vector<float3> vertices;
    std::vector<uint3> triangles;
    std::vector<float3> normals;
    construct_primitives(data, vertices, triangles, normals);

    auto toDevice = [&](auto& dst, void* src, size_t size){
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dst), size));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dst), src, size, cudaMemcpyHostToDevice));
    };
    
    toDevice(d_vertices, vertices.data(), vertices.size() * sizeof(float3));
    toDevice(d_triangles, triangles.data(), triangles.size() * sizeof(uint3));

    d_gaussians.numgs = data.numgs;
    toDevice(d_gaussians.xyz, data.xyz, data.numgs*sizeof(float3));
    toDevice(d_gaussians.rotation, data.rotation, data.numgs*sizeof(float4));
    toDevice(d_gaussians.scaling, data.scaling, data.numgs*sizeof(float3));
    toDevice(d_gaussians.opacity, data.opacity, data.numgs*sizeof(float));
    toDevice(d_gaussians.sh, data.sh, data.numgs*sizeof(float3)*16);
    toDevice(d_gaussians.normals, normals.data(), normals.size()*sizeof(float3));

    // Allocate buffers of scene size 
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_scene_buffers.rad_clamped), data.numgs*3*sizeof(bool)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_scene_buffers.rad_sh), data.numgs*sizeof(float3)));

    // Use default options for simplicity.  In a real use case we would want to
    // enable compaction, etc
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Our build input is a simple list of non-indexed triangle vertices
    const uint32_t triangle_input_flags[1] ={OPTIX_GEOMETRY_FLAG_NONE};// {OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING}; - does not do anything
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexBuffers = &d_vertices;
    triangle_input.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());

    triangle_input.triangleArray.indexFormat = OptixIndicesFormat::OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexBuffer = d_triangles;
    triangle_input.triangleArray.numIndexTriplets = static_cast<uint32_t>(triangles.size());

    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        context,
        &accel_options,
        &triangle_input,
        1,  // Number of build inputs
        &gas_buffer_sizes));
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_temp_buffer_gas),
        gas_buffer_sizes.tempSizeInBytes));

    //std::cout << "GAS size " << gas_buffer_sizes.outputSizeInBytes/1024.f/1024.f << " MiB" << std::endl;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void **>(&d_gas_output_buffer),
        gas_buffer_sizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(
        context,
        0,  // CUDA stream
        &accel_options,
        &triangle_input,
        1,  // num build inputs
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        d_gas_output_buffer,
        gas_buffer_sizes.outputSizeInBytes,
        &gas_handle_,
        nullptr,  // emitted property list
        0         // num emitted properties
        ));

    // We can now free the scratch space buffer used during build and the vertex
    // inputs, since they are not needed by our trivial shading method
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer_gas)));
}
