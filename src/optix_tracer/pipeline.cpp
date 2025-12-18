#include "pipeline.h"

#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <cassert>
#include <iostream>
#include "utils/exception.h"
#include "optix_types.h"

using namespace gsrt::optix_tracer;

// These structs represent the data blocks of our SBT records
template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;


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
        constexpr int payload_size_fwd = 32;//8;
        unsigned int semantics_fwd[payload_size_fwd];/* = {
                                    payload_flags,
                                    payload_flags,
                                    payload_flags,
                                    payload_flags,

                                    payload_flags,
                                    payload_flags,

                                    payload_flags,

                                    payload_flags,
        };*/
        for(int i = 0; i < 32; ++i){
            semantics_fwd[i] = payload_flags;
        }
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
        pipeline_compile_options.usesPrimitiveTypeFlags |= OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

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
            
#ifdef RG_BWD
        OptixProgramGroupDesc raygen_bwd_prog_group_desc = {};
        raygen_bwd_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_bwd_prog_group_desc.raygen.module = module;
        raygen_bwd_prog_group_desc.raygen.entryFunctionName = "__raygen__bwd";
        OPTIX_CHECK_LOG(optixProgramGroupCreate(
            context,
            &raygen_bwd_prog_group_desc,
            1,  // num program groups
            &empty_program_group_options,
            log,
            &sizeof_log,
            &raygen_bwd_prog_group));
#endif

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
        //hitgroup_prog_group_desc.hitgroup.moduleIS = module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__fwd";
        //hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__is";
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
        //bwd_hitgroup_prog_group_desc.hitgroup.moduleIS = module;
        bwd_hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = nullptr;
        bwd_hitgroup_prog_group_desc.hitgroup.entryFunctionNameAH = "__anyhit__bwd";
        //bwd_hitgroup_prog_group_desc.hitgroup.entryFunctionNameIS = "__intersection__is";
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
        rg_fwd = raygen_record;

#ifdef RG_BWD
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&rg_bwd), raygen_record_size));
        OPTIX_CHECK(optixSbtRecordPackHeader(raygen_bwd_prog_group, &rg_sbt));
        CUDA_CHECK(cudaMemcpy(
            reinterpret_cast<void *>(rg_bwd),
            &rg_sbt,
            raygen_record_size,
            cudaMemcpyHostToDevice));
#endif

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
      rg_fwd(std::exchange(other.rg_fwd, {})),
      rg_bwd(std::exchange(other.rg_bwd, {})),
      stream(std::exchange(other.stream, nullptr)),
      d_param(std::exchange(other.d_param, 0)) {}
      


TraceRaysPipeline::~TraceRaysPipeline(){
    const auto device = std::exchange(this->device, -1);
    if (device == -1) {
        return;
    }
    CUDA_CHECK(cudaSetDevice(device));
    if (d_param != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(d_param, 0))));
    //if (sbt.raygenRecord != 0)
    //    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(sbt.raygenRecord, 0))));
    if (rg_fwd != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(rg_fwd, 0))));
    if (rg_bwd != 0)
        CUDA_CHECK(cudaFree(reinterpret_cast<void *>(std::exchange(rg_bwd, 0))));
    //------------
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
    if (raygen_bwd_prog_group != nullptr)
        OPTIX_CHECK(optixProgramGroupDestroy(std::exchange(raygen_bwd_prog_group, nullptr)));
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
    Params params;
    int width, height;
    params.handle = gaussians_structure->gas_handle();
    {
        const auto& r = tracing_params.rays;
        params.ray_origins = r.ray_origins;
        params.ray_directions = r.ray_directions;
        width = r.width;
        height = r.height;
    }
    auto& gs = gaussians_structure->gaussians();
    params.num_gs = gs.numgs;
    params.gs_xyz = gs.xyz;
    params.gs_rotation = gs.rotation;
    params.gs_scaling = gs.scaling;
    params.gs_opacity = gs.opacity;
    params.gs_sh = gs.sh;
    params.sh_deg = gs.sh_deg;
    //params.gs_color = gs.color;
    params.gs_normals = gaussians_structure->normals();
    {
        auto s = std::get<OptixRenderSettings>(tracing_params.settings);
        params.compute_grad = s.compute_grad;
        params.white_background = s.white_background;
    }
    {
        const auto& o = tracing_params.output;
        params.radiance = o.radiance;
        params.transmittance = o.transmittance;
        params.debug_map_0 = o.debug_map_0;
        params.debug_map_1 = o.debug_map_1;
        params.num_its = o.num_its;
        params.num_its_bwd = o.num_its_bwd;
        params.distance = o.distance;
    }
    if (std::get<OptixRenderSettings>(tracing_params.settings).compute_grad) {
        assert(tracing_params.bp_out.has_value());
        const auto& bp_out = tracing_params.bp_out.value();
        params.grad_xyz = bp_out.grad_xyz;
        params.grad_rotation = bp_out.grad_rotation;
        params.grad_scale = bp_out.grad_scale;
        params.grad_opacity = bp_out.grad_opacity;
        params.grad_sh = bp_out.grad_sh;
        //params.grad_resp = bp_out.grad_resp;
        //params.grad_color = bp_out.grad_color;
        //params.grad_invRS = bp_out.grad_invRS;
        assert(tracing_params.bp_in.has_value());
        const auto& bp_in = tracing_params.bp_in.value();
        params.dL_dC = bp_in.dL_dC;
        params.radiance = bp_in.radiance;
        params.transmittance = bp_in.transmittance;
        params.distance = bp_in.distance;
    }
    if(params.compute_grad && rg_bwd != 0){
        sbt.raygenRecord = rg_bwd;
    }else{
        sbt.raygenRecord = rg_fwd;
    }
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void *>(d_param),
        &params, sizeof(params),
        cudaMemcpyHostToDevice));
    OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(params), &sbt, width, height, 1));
    //OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(params), &sbt,
    //                        tracing_params.num_rays, 1, 1));
    CUDA_SYNC_CHECK();
    CUDA_CHECK(cudaStreamSynchronize(stream));
}
