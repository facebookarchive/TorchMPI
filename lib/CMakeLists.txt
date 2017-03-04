set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${MPI_CXX_COMPILE_FLAGS})
set(CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS})
include_directories(${MPI_CXX_INCLUDE_PATH})

find_package(CUDA)

if(CUDA_FOUND)
  message(STATUS "CUDA Found")
  include_directories(${CUDA_INCLUDE_DIRS})
  include(CheckLibraryExists)
#  check_library_exists(THC THCudaInit "" WITH_CUDA)
  find_library(THC NAMES libTHC.so "" WITH_CUDA)
  if(THC)
    set(WITH_CUDA 1)
  endif()
  if(NOT WITH_CUDA)
    message(WARNING "CUDA found, but THC library not found -- disabling CUDA support")
  else()
    set(CMAKE_REQUIRED_INCLUDES ${CUDA_INCLUDE_DIRS})
    include(CheckIncludeFile)
    check_include_file(nccl.h WITH_NCCL)
    if(WITH_NCCL)
      add_definitions(-DTORCH_MPI_NCCL)
    else()
      message(WARNING "NCCL not found -- please install it yourself if you need it")
    endif()
  endif()
endif()

set(SRC
  barrier.cpp
  collectives.cpp
  constants.cpp
  parameterserver.cpp
  resources.cpp
  torch_mpi.cpp
  detail/collectives.cpp)

if(WITH_CUDA)
   add_definitions(-DTORCH_MPI_CUDA)
   INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/cmake/select_compute_arch.cmake)
   CUDA_SELECT_NVCC_ARCH_FLAGS(NVCC_FLAGS_EXTRA)
   set (CUDA_VERBOSE_BUILD ON)
   set (CUDA_PROPAGATE_HOST_FLAGS ON)
   include_directories(${Torch_INSTALL_INCLUDE}/THC)
   add_definitions(-DTORCH_MPI_CUDA)
   set(SRC
     ${SRC}
     collectives_cuda.cpp
     torch_mpi_cuda.cpp
     detail/collectives_cuda.cpp
     detail/reduce_kernel.cu)
endif()

if(WITH_CUDA)
  cuda_add_library(torchmpi MODULE ${SRC} OPTIONS -std c++11) # module for mac os x support
#DEBUG:  target_link_libraries(torchmpi TH THC)
else()
  add_library(torchmpi MODULE ${SRC}) # module for mac os x support
  target_link_libraries(torchmpi TH)
endif()
target_link_libraries(torchmpi ${MPI_CXX_LIBRARIES})

if(WITH_NCCL)
  target_link_libraries(torchmpi nccl_static)
endif()

find_library(GLOO_LIBRARIES gloo)
find_path(GLOO_INCLUDE_DIR gloo/context.h)
if (GLOO_LIBRARIES AND GLOO_INCLUDE_DIR)
  set(WITH_GLOO 1)
endif()
if(WITH_CUDA)
  find_library(GLOO_CUDA_LIBRARIES gloo_cuda)
endif()

if(WITH_GLOO)
  message(STATUS "GLOO found")
  add_definitions(-DTORCH_MPI_GLOO)
  include_directories(${GLOO_INCLUDE_DIR})
  target_link_libraries(torchmpi ${GLOO_LIBRARIES})
  if (GLOO_CUDA_LIBRARIES)
    message(STATUS "GLOO cuda library found")
    add_definitions(-DTORCH_MPI_GLOO_CUDA)
    target_link_libraries(torchmpi ${GLOO_CUDA_LIBRARIES})
  else()
    message(WARNING "GLOO cuda library not found -- please install it yourself if you need it")
  endif()
else()
    message(WARNING "GLOO not found -- please install it yourself if you need it")
endif()

install(TARGETS torchmpi DESTINATION ${LUA_CPATH})

### static lib, if needed
# if(WITH_CUDA)
#   cuda_add_library(torchmpi-static STATIC ${SRC}) # module for mac os x support
# else()
#   add_library(torchmpi-static STATIC ${SRC}) # module for mac os x support
# endif()
# target_link_libraries(torchmpi-static ${MPI_CXX_LIBRARIES})
# if(WITH_NCCL)
#   target_link_libraries(torchmpi-static nccl_static)
# endif()
# if (WITH_GLOO)
#   target_link_libraries(torchmpi-static ${GLOO_LIBRARIES})
#   if (GLOO_CUDA_LIBRARIES)
#     target_link_libraries(torchmpi-static ${GLOO_CUDA_LIBRARIES})
#   endif()
# endif()
