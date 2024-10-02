#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include <iostream>
#include <memory>
#include <string>

#include "gaussians_tracer.h"
#include "utils/exception.h"

namespace py = pybind11;
using namespace pybind11::literals;  // to bring in the `_a` literal

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_DEVICE(x) TORCH_CHECK(x.device() == this->device, #x " must be on the same device")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " must have float32 type")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT_DIM(x,d) \
    CHECK_INPUT(x);         \
    CHECK_DEVICE(x);        \
    CHECK_FLOAT(x);         \
    TORCH_CHECK(x.size(-1) == d, #x " must have last dimension with size d")
#define CHECK_HOST_FLOAT_DIM(x,d) \
    CHECK_CONTIGUOUS(x)     \
    CHECK_FLOAT(x);         \
    TORCH_CHECK(x.size(-1) == d, #x " must have last dimension with size d")


struct PyGaussiansTracer {
    PyGaussiansTracer(const torch::Device &device) : device(device) {
        if (!device.is_cuda()) {
            throw Exception("The device argument must be a CUDA device.");
        }
        tracer = std::make_unique<GaussiansTracer>(device.index());
    }
    ~PyGaussiansTracer() {
        tracer.reset();
    }

    py::dict trace_rays(const torch::Tensor &ray_origins,
                        const torch::Tensor &ray_directions) {

        torch::AutoGradMode enable_grad(false);
        CHECK_FLOAT_DIM(ray_origins,3);
        CHECK_FLOAT_DIM(ray_directions,3);
        const size_t num_rays = ray_origins.numel() / 3;
        const auto radiance = torch::zeros({(long)num_rays, 3}, torch::device(device).dtype(torch::kFloat32));
        const auto transmittance = torch::zeros({(long)num_rays}, torch::device(device).dtype(torch::kFloat32));

        const auto debug_map_0 = torch::zeros({(long)num_rays, 3}, torch::device(device).dtype(torch::kFloat32));
        const auto debug_map_1 = torch::zeros({(long)num_rays, 3}, torch::device(device).dtype(torch::kFloat32));

        tracer->trace_rays(
            num_rays,
            reinterpret_cast<float3 *>(ray_origins.data_ptr()),
            reinterpret_cast<float3 *>(ray_directions.data_ptr()),
            reinterpret_cast<float3 *>(radiance.data_ptr()),
            reinterpret_cast<float *>(transmittance.data_ptr()),
            reinterpret_cast<float3 *>(debug_map_0.data_ptr()),
            reinterpret_cast<float3 *>(debug_map_1.data_ptr())
            );

        return py::dict("radiance"_a = radiance, "transmittance"_a = transmittance,
                        "debug_map_0"_a = debug_map_0, "debug_map_1"_a = debug_map_1);
    }

    void load_gaussians(
        const torch::Tensor &xyz,
        const torch::Tensor &rotation,
        const torch::Tensor &scaling,
        const torch::Tensor &opacity,
        const torch::Tensor &sh,
        const int sh_deg) {

        CHECK_HOST_FLOAT_DIM(xyz,3);
        CHECK_HOST_FLOAT_DIM(rotation,4);
        CHECK_HOST_FLOAT_DIM(scaling,3);
        CHECK_HOST_FLOAT_DIM(opacity,1);
        CHECK_HOST_FLOAT_DIM(sh,3);
        
        GaussiansData data;
        data.numgs = xyz.numel() / 3;
        data.xyz = reinterpret_cast<float3 *>(xyz.data_ptr());
        data.rotation = reinterpret_cast<float4 *>(rotation.data_ptr());
        data.scaling = reinterpret_cast<float3 *>(scaling.data_ptr());
        data.opacity = reinterpret_cast<float *>(opacity.data_ptr());
        data.sh = reinterpret_cast<float3 *>(sh.data_ptr());
        data.sh_deg = sh_deg;
        tracer->load_gaussians(data);
    }

    const torch::Device &get_device() const {
        return this->device;
    }

   private:
    std::unique_ptr<GaussiansTracer> tracer;
    torch::Device device;
};

PYBIND11_MODULE(gsrt_cpp_extension, m) {
    py::class_<PyGaussiansTracer>(m, "GaussiansTracer")
        .def(py::init<const torch::Device &>())
        .def_property_readonly("device", &PyGaussiansTracer::get_device)
        .def("trace_rays", &PyGaussiansTracer::trace_rays)
        .def("load_gaussians", &PyGaussiansTracer::load_gaussians);
}
