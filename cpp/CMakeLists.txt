cmake_minimum_required(VERSION 3.16)
project(gmr_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(GMR_THIRDPARTY_PREFIX "/opt/robot/devel" CACHE PATH "Third-party install prefix")
set(GMR_MUJOCO_PREFIX "/opt/robot/devel" CACHE PATH "MuJoCo install prefix")
list(PREPEND CMAKE_PREFIX_PATH "${GMR_THIRDPARTY_PREFIX}" "${GMR_MUJOCO_PREFIX}")
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/devel-packages")

set(CMAKE_BUILD_RPATH "${GMR_THIRDPARTY_PREFIX}/lib;${GMR_MUJOCO_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH "${GMR_THIRDPARTY_PREFIX}/lib;${GMR_MUJOCO_PREFIX}/lib")

include(FetchContent)
set(DAQP_PROFILING OFF CACHE BOOL "" FORCE)
set(DAQP_TEST OFF CACHE BOOL "" FORCE)
set(DAQP_BENCHMARK OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  daqp
  GIT_REPOSITORY https://github.com/darnstrom/daqp.git
  GIT_TAG master
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(daqp)

find_package(glog REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(qpOASES REQUIRED)
find_package(pinocchio REQUIRED)
find_package(mujoco REQUIRED)

find_package(OpenMP)

if(NOT TARGET qpOASES::qpOASES)
  add_library(qpOASES::qpOASES UNKNOWN IMPORTED)
  set_target_properties(qpOASES::qpOASES PROPERTIES
    IMPORTED_LOCATION "${qpOASES_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${qpOASES_INCLUDE_DIR}")
endif()
find_package(glfw3 QUIET)
if(NOT glfw3_FOUND)
  find_library(GLFW_LIBRARY NAMES glfw REQUIRED)
endif()

find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
  find_path(YAML_CPP_INCLUDE_DIR
    NAMES yaml-cpp/yaml.h
    PATHS "${GMR_THIRDPARTY_PREFIX}/include" "${GMR_MUJOCO_PREFIX}/include"
    REQUIRED)
  find_library(YAML_CPP_LIBRARY
    NAMES yaml-cpp
    PATHS "${GMR_THIRDPARTY_PREFIX}/lib" "${GMR_MUJOCO_PREFIX}/lib"
    REQUIRED)
endif()

add_library(gmr_retarget_core STATIC
  src/solver/qp_solver.cpp
  src/solver/hqp_solver.cpp
  src/retarget/ik_config.cpp
  src/retarget/contact_ground_config.cpp
  src/retarget/contact_ground.cpp
  src/retarget/mujoco_collision_limit.cpp
  src/retarget/human_frame_io.cpp
  src/retarget/robot_trajectory_pipeline.cpp
  src/retarget/retargeter.cpp
  src/retarget/retargeter_pinocchio.cpp
  src/retarget/retargeter_mujoco.cpp
  src/retarget/batch_trajectory_retarget.cpp
  src/retarget/online_qp_config.cpp
  src/retarget/online_qp_retarget.cpp
)

target_include_directories(gmr_retarget_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${GMR_THIRDPARTY_PREFIX}/include
  ${GMR_MUJOCO_PREFIX}/include
  ${daqp_SOURCE_DIR}/include
)

target_link_libraries(gmr_retarget_core PUBLIC
  Eigen3::Eigen
  qpOASES::qpOASES
  daqpstat
  pinocchio::pinocchio
  mujoco::mujoco
  glog::glog
)

if(yaml-cpp_FOUND)
  if(TARGET yaml-cpp::yaml-cpp)
    target_link_libraries(gmr_retarget_core PUBLIC yaml-cpp::yaml-cpp)
  elseif(TARGET yaml-cpp)
    target_link_libraries(gmr_retarget_core PUBLIC yaml-cpp)
  else()
    target_link_libraries(gmr_retarget_core PUBLIC yaml-cpp)
  endif()
else()
  target_include_directories(gmr_retarget_core PUBLIC ${YAML_CPP_INCLUDE_DIR})
  target_link_libraries(gmr_retarget_core PUBLIC ${YAML_CPP_LIBRARY})
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(gmr_retarget_core PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(gmr_retarget_cli
  src/main_retarget_cli.cpp
)

target_link_libraries(gmr_retarget_cli PRIVATE
  gmr_retarget_core
)

add_executable(gmr_retarget_batch_cli
  src/main_retarget_batch_cli.cpp
)

target_link_libraries(gmr_retarget_batch_cli PRIVATE
  gmr_retarget_core
)

add_executable(gmr_batch_to_cli
  src/main_batch_to_cli.cpp
)

target_link_libraries(gmr_batch_to_cli PRIVATE
  gmr_retarget_core
)

add_executable(gmr_online_qp_cli
  src/main_online_qp_cli.cpp
)

add_executable(gmr_robot_to_robot_cli
  src/main_robot_trajectory_to_gmr.cpp
)

target_link_libraries(gmr_robot_to_robot_cli PRIVATE
  gmr_retarget_core
)

target_link_libraries(gmr_online_qp_cli PRIVATE
  gmr_retarget_core
)

add_executable(gmr_retarget_viewer
  src/main_retarget_viewer.cpp
)

if(glfw3_FOUND)
  target_link_libraries(gmr_retarget_viewer PRIVATE
    gmr_retarget_core
    mujoco::mujoco
    glfw
  )
else()
  target_link_libraries(gmr_retarget_viewer PRIVATE
    gmr_retarget_core
    mujoco::mujoco
    ${GLFW_LIBRARY}
  )
endif()
