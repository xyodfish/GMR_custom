cmake_minimum_required(VERSION 3.16)
project(gmr_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(GMR_THIRDPARTY_PREFIX "/opt/galbot/devel/x86_64-Linux-GNU-9.4.0" CACHE PATH "Third-party install prefix")
set(GMR_MUJOCO_PREFIX "/opt/galbot/devel_control/x86_64-Linux-GNU-9.4.0" CACHE PATH "MuJoCo install prefix")
list(PREPEND CMAKE_PREFIX_PATH "${GMR_THIRDPARTY_PREFIX}")
list(PREPEND CMAKE_PREFIX_PATH "${GMR_MUJOCO_PREFIX}")

set(CMAKE_BUILD_RPATH "${GMR_THIRDPARTY_PREFIX}/lib;${GMR_MUJOCO_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH "${GMR_THIRDPARTY_PREFIX}/lib;${GMR_MUJOCO_PREFIX}/lib")

find_package(Eigen3 REQUIRED)
find_package(qpOASES REQUIRED)
find_package(mujoco REQUIRED)
find_package(glfw3 QUIET)
if(NOT glfw3_FOUND)
  find_library(GLFW_LIBRARY NAMES glfw REQUIRED)
endif()

add_library(gmr_retarget_core STATIC
  src/solver/qp_solver.cpp
  src/solver/hqp_solver.cpp
  src/retarget/ik_config.cpp
  src/retarget/human_frame_io.cpp
  src/retarget/retargeter.cpp
)

target_include_directories(gmr_retarget_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${GMR_THIRDPARTY_PREFIX}/include
  ${GMR_MUJOCO_PREFIX}/include
)

target_link_libraries(gmr_retarget_core PUBLIC
  Eigen3::Eigen
  qpOASES
  mujoco::mujoco
)

add_executable(gmr_retarget_cli
  src/main_retarget_cli.cpp
)

target_link_libraries(gmr_retarget_cli PRIVATE
  gmr_retarget_core
)

add_executable(gmr_retarget_viewer
  src/main_retarget_viewer.cpp
)

if(glfw3_FOUND)
  target_link_libraries(gmr_retarget_viewer PRIVATE
    gmr_retarget_core
    glfw
  )
else()
  target_link_libraries(gmr_retarget_viewer PRIVATE
    gmr_retarget_core
    ${GLFW_LIBRARY}
  )
endif()
