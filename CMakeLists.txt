set(CMAKE_BUILD_TYPE Release)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

if(CMAKE_VERSION VERSION_LESS "3.1")
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
  if(NOT COMPILER_SUPPORTS_CXX11)
    MESSAGE(FATAL_ERROR "Your compiler does not support c++11")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  set(CMAKE_CXX_STANDARD 11)
endif()

find_package(Torch REQUIRED)
find_package(MPI)
if(NOT MPI_CXX_FOUND)
  MESSAGE(FATAL_ERROR "MPI (c++) not found; please set MPI_CXX_COMPILER env variable")
endif()

set(PKGNAME torchmpi)

add_subdirectory(lib) # C++ lib
add_subdirectory(torchmpi) # lua lib
