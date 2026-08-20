cmake_minimum_required(VERSION 3.16)
project(wbc_rt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED ENV{CMEEL_PREFIX} AND EXISTS "/media/camp/EXT_DRIVE/envs/rm75/lib/python3.10/site-packages/cmeel.prefix")
  set(ENV{CMEEL_PREFIX} "/media/camp/EXT_DRIVE/envs/rm75/lib/python3.10/site-packages/cmeel.prefix")
endif()
if(DEFINED ENV{CMEEL_PREFIX})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{CMEEL_PREFIX}")
endif()

find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(coal REQUIRED)

# proxsuite CMake config requires Simde via a custom Find module that is not
# installed here.  The library is header-only; include it directly.
set(SIMDE_HINTS
  "${CMAKE_CURRENT_SOURCE_DIR}/third_party/simde"
  "$ENV{CMEEL_PREFIX}/include"
  "/usr/include"
)
find_path(SIMDE_INCLUDE_DIR simde/simde-math.h HINTS ${SIMDE_HINTS})
if(NOT SIMDE_INCLUDE_DIR)
  message(FATAL_ERROR
    "simde headers not found. Run native/wbc_rt/build.sh (downloads third_party/simde).")
endif()

add_library(simde INTERFACE)
target_include_directories(simde INTERFACE "${SIMDE_INCLUDE_DIR}")

add_library(proxsuite_headers INTERFACE)
target_include_directories(proxsuite_headers INTERFACE "$ENV{CMEEL_PREFIX}/include")
target_link_libraries(proxsuite_headers INTERFACE Eigen3::Eigen simde)

add_executable(wbc_rt
  src/main.cpp
  src/config.cpp
  src/kinematics.cpp
  src/inner.cpp
  src/srs_ik.cpp
  src/posture.cpp
)
target_include_directories(wbc_rt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(wbc_rt PRIVATE
  Eigen3::Eigen
  pinocchio::pinocchio
  pinocchio::pinocchio_parsers
  pinocchio::pinocchio_collision
  proxsuite_headers
  coal::coal
  rt
  pthread
)

install(TARGETS wbc_rt RUNTIME DESTINATION bin)
