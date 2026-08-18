cmake_minimum_required(VERSION 3.16)
project(qpik_kernel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED ENV{CMEEL_PREFIX} AND EXISTS "/media/camp/EXT_DRIVE/envs/rm75/lib/python3.10/site-packages/cmeel.prefix")
  set(ENV{CMEEL_PREFIX} "/media/camp/EXT_DRIVE/envs/rm75/lib/python3.10/site-packages/cmeel.prefix")
endif()
if(DEFINED ENV{CMEEL_PREFIX})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{CMEEL_PREFIX}")
endif()

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

pybind11_add_module(_qpik_kernel src/qpik_kernel.cpp NO_EXTRAS)
target_link_libraries(_qpik_kernel PRIVATE Eigen3::Eigen)
target_compile_definitions(_qpik_kernel PRIVATE QPIK_KERNEL_VERSION=1)
set_target_properties(_qpik_kernel PROPERTIES INTERPROCEDURAL_OPTIMIZATION OFF)

install(TARGETS _qpik_kernel
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}
)
