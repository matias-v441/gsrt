#include "as.h"
#include <optix_stubs.h>
#include "utils/exception.h"
#include "utils/primitives.h"
using namespace gsrt::optix_tracer;

GaussiansAS::GaussiansAS() noexcept
    : device(-1),
      context(nullptr),
      gas_handle_(0),
      d_gas_output_buffer(0),
      d_vertices(0),
      d_triangles(0),
      d_normals(0) {}


GaussiansAS::GaussiansAS(GaussiansAS &&other) noexcept
    : device(std::exchange(other.device, -1)),
      context(std::exchange(other.context, nullptr)),
      gas_handle_(std::exchange(other.gas_handle_, 0)),
      d_gas_output_buffer(std::exchange(other.d_gas_output_buffer, 0)),
      d_vertices(std::exchange(other.d_vertices, 0)),
      d_triangles(std::exchange(other.d_triangles, 0)),
      d_normals(std::exchange(other.d_normals, nullptr)),
      d_gaussians(std::exchange(other.d_gaussians, {})),
      _gas_size(std::exchange(other._gas_size, {})){}

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
    device_free(d_normals);
}

GaussiansAS::~GaussiansAS(){
    if (this->device != -1) {
        release();
    }
    const auto device = std::exchange(this->device, -1);
}

void GaussiansAS::build() {
    release();

    CUDA_CHECK(cudaSetDevice(device));

    uint32_t nvert,ntriag;
    {
    using namespace util::geom::cuda;
    alloc_buffers(d_gaussians.numgs,reinterpret_cast<float3**>(&d_vertices),nvert,reinterpret_cast<uint3**>(&d_triangles),ntriag);
    construct_icosahedra(d_gaussians.numgs,d_gaussians.xyz,d_gaussians.opacity,d_gaussians.scaling,d_gaussians.rotation,
        reinterpret_cast<float3*>(d_vertices), reinterpret_cast<uint3*>(d_triangles));
    }

    // assert(svrt/3 == nvert);
    // assert(stri/3 == ntriag);
    // d_vertices = reinterpret_cast<CUdeviceptr>(vrt);
    // d_triangles = reinterpret_cast<CUdeviceptr>(tri);

    // Use default options for simplicity.  In a real use case we would want to
    // enable compaction, etc
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    //accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
    //accel_options.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Our build input is a simple list of non-indexed triangle vertices
    const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL};
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexBuffers = &d_vertices;
    triangle_input.triangleArray.numVertices = nvert;//static_cast<uint32_t>(vertices.size());

    triangle_input.triangleArray.indexFormat = OptixIndicesFormat::OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexBuffer = d_triangles;
    triangle_input.triangleArray.numIndexTriplets = ntriag; //static_cast<uint32_t>(triangles.size());

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

    _gas_size = gas_buffer_sizes.outputSizeInBytes;

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
