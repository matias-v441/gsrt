#pragma once
#include <optix.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include "types.h"

namespace gsrt::optix_tracer {

class GaussiansAS {
   public:
    GaussiansAS() noexcept;
    GaussiansAS(const OptixDeviceContext &context, const uint8_t device) : device(device), context(context) {}
    GaussiansAS(
        const OptixDeviceContext &context,
        const uint8_t device,
        const GaussiansData& d_gaussians):GaussiansAS(context, device){
        this->d_gaussians = d_gaussians;
        build();
    }

    ~GaussiansAS();
    GaussiansAS(const GaussiansAS &) = delete;
    GaussiansAS &operator=(const GaussiansAS &) = delete;
    GaussiansAS(GaussiansAS &&other) noexcept;
    GaussiansAS &operator=(GaussiansAS &&other) {
        using std::swap;
        if (this != &other) {
            GaussiansAS tmp(std::move(other));
            swap(tmp, *this);
        }
        return *this;
    }
    friend void swap(GaussiansAS &first, GaussiansAS &second) {
        using std::swap;
        swap(first.context, second.context);
        swap(first.device, second.device);
        swap(first.gas_handle_, second.gas_handle_);
        swap(first.d_gas_output_buffer, second.d_gas_output_buffer);
        swap(first.d_vertices, second.d_vertices);
        swap(first.d_triangles, second.d_triangles);
        swap(first.d_normals, second.d_normals);
        swap(first.d_gaussians, second.d_gaussians);
        swap(first._gas_size, second._gas_size);
    }

    OptixTraversableHandle gas_handle() const {
        if (!defined()) {
            throw std::runtime_error("TetrahedraStructure is not initialized");
        }
        return gas_handle_;
    }

    bool defined() const {
        return gas_handle_ != 0;
    }

    const GaussiansData& gaussians() const{
        return d_gaussians;
    }

    const float3* normals() const{
        return d_normals;
    }

    const size_t gas_size() const{
        return _gas_size;
    }

   private:
    void build();

    void release();
    OptixDeviceContext context = nullptr;
    int8_t device = -1;
    OptixTraversableHandle gas_handle_ = 0;
    CUdeviceptr d_gas_output_buffer = 0;
    CUdeviceptr d_vertices = 0;
    CUdeviceptr d_triangles = 0;
    float3* d_normals = 0;
    GaussiansData d_gaussians{};
    size_t _gas_size{};
};
}  // namespace gsrt::optix_tracer::as