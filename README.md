# Ray tracing 3D Gaussians
The project was aimed to implement the method proposed in 3DGRT paper by NVIDIA before any public implementation existed. 
The initial goal though was identified as rendering scenes trained by gaussian splatting. It turned out that the rendering algorithm
proposed in the paper produced significant artifacts for this case due to k-buffer being too small, so a different approach was taken to match the reference.

The optimization part was not done correctly, and I have run out of time trying to fix it.
