# Ray tracing 3D Gaussians
A minimal implementation of 3DGRT with additional MLAB sorting version of the tracer.

## Installation 
This uses CUDA 12.5 and OptiX 8. The simple-knn module has to be compiled with CUDA 11. For compilation please see the setup script. To change versions of the tracer see CMake. This is the development version, so the user experience is not included. For fisheye cameras and training visualization please use the attached nerfbaselines version (yeah, I know). In the same way I add my version of FisheyeGS under nbs-3dgs-fs/.

![splatting](renders/DSC07112-splat.png)
![tracing](renders/DSC07112-trace.png)