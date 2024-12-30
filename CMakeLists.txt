cmake_minimum_required(VERSION 3.22.1)
project(fast_traversal LANGUAGES CXX CUDA)
enable_language(CUDA)

include(FetchContent)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_BUILD_TYPE "Debug")

include_directories(${CONDA_PREFIX}/include)
link_directories(${CONDA_PREFIX}/lib)
message("CONDA_PREFIX ${CONDA_PREFIX}")
find_library(LIBCNPY cnpy ${CONDA_PREFIX}/lib)
message("LIBCNPY ${LIBCNPY}")

find_package(Torch REQUIRED)
find_package(CGAL REQUIRED)

find_package(glm CONFIG REQUIRED)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.9.2
)
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

FetchContent_Declare(
    cnpy
    GIT_REPOSITORY https://github.com/rogersce/cnpy
)
FetchContent_GetProperties(cnpy)
if(NOT cnpy_POPULATED)
    FetchContent_Populate(cnpy)
    add_subdirectory(${cnpy_SOURCE_DIR} ${cnpy_BINARY_DIR})
endif()

# Put all the runtime stuff in the same directory.  By default, CMake puts each targets'
# output into their own directory.  We want all the targets to be put in the same
# directory, and we can do this by setting these variables.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(BUILD_DIRECTORY "${CMAKE_BINARY_DIR}/build")
set(CUDA_GENERATED_OUTPUT_DIR "${BUILD_DIRECTORY}")

# Locate the NVRT distribution.  Search the SDK first, then look in the system.
set(OptiX_INSTALL_DIR "/opt/optix" CACHE PATH "Path to OptiX installed location.")
if(NOT "$ENV{OPTIX_PATH}" STREQUAL "")
    set(OptiX_INSTALL_DIR "$ENV{OPTIX_PATH}")
endif()

# Search for the OptiX libraries and include files.
find_package(OptiX REQUIRED)
# Add the path to the OptiX headers to our include paths.
include_directories(
  "${OptiX_INCLUDE}"
  "${CMAKE_CURRENT_SOURCE_DIR}/cuda"
)

message("OptiX: ${OptiX_INCLUDE}")

find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
include_directories(${OptiX_INCLUDE} ${CUDA_INCLUDE_DIRS})

find_program(BIN2C bin2c
  DOC "Path to the cuda-sdk bin2c executable.")


macro(cuda_compile_and_embed output_var generated_files_name)
    set(c_var_name ${output_var})
    set(generated_files "${${generated_files_name}}")
    list(GET "${generated_files_name}" 0 ptx_file)
    get_filename_component(embedded_file ${ptx_file}_embedded.cpp NAME)
    set(embedded_file "${BUILD_DIRECTORY}/${embedded_file}")
    add_custom_command(
        OUTPUT ${embedded_file}
        COMMAND ${BIN2C} --padd 0 --type char --name ${c_var_name} ${ptx_file} > ${embedded_file}
        DEPENDS ${ptx_files} ${generated_files}
        COMMENT "compiling (and embedding ptx from) ${cuda_file}")
    set(${output_var} ${embedded_file})
    list(APPEND "${generated_files_name}" ${embedded_file})
endmacro()

CUDA_WRAP_SRCS(traversal PTX gaussians_tracer_ptx
    src/optix_types.h
    src/utils/preprocessor.h
    src/utils/vec_math.h
    #src/gaussians_tracer.cu
    #src/gs_tracer_sorting.cu
    #src/gs_tracer_sorting_2d.cu
    #src/gs_tracer_sorting_2d_queue.cu
    src/gs_tracer_bwd.cu
    #src/gs_tracer_sorting_2d_merge.cu
    #src/gs_tracer_chit_2d.cu
    OPTIONS -rdc true)
cuda_compile_and_embed(ptx_code_file gaussians_tracer_ptx)

add_link_options(-flto=auto)  # disables some warning
pybind11_add_module(gsrt_cpp_extension
    src/gaussians_tracer.cpp
    src/tracer_custom.cpp
    src/py_binding.cpp
    src/tracer_cu.cu
    ${gaussians_tracer_ptx}
    )

target_link_libraries(gsrt_cpp_extension PRIVATE ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})
target_link_libraries(gsrt_cpp_extension PRIVATE CGAL::CGAL)
target_link_libraries(gsrt_cpp_extension PRIVATE glm::glm)
target_compile_features(gsrt_cpp_extension PRIVATE cxx_std_17)
set_target_properties(gsrt_cpp_extension
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/extension"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

add_executable(test_cpp test.cpp src/gaussians_tracer.cpp ${gaussians_tracer_ptx})
target_include_directories(test_cpp PRIVATE src)
#target_link_libraries(test_cpp PRIVATE ${LIBCNPY})
target_link_libraries(test_cpp PRIVATE cnpy)
target_link_libraries(test_cpp PRIVATE ${CUDA_LIBRARIES} ${TORCH_LIBRARIES})
target_link_libraries(test_cpp PRIVATE CGAL::CGAL)
target_compile_features(test_cpp PRIVATE cxx_std_17)
