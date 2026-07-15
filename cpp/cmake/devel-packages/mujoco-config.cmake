# Minimal mujoco CMake package for /opt/robot/devel installs.
get_filename_component(_MUJOCO_PREFIX "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

set(mujoco_VERSION "3.10.0")
set(mujoco_INCLUDE_DIR "${_MUJOCO_PREFIX}/include")
set(mujoco_LIBRARY "${_MUJOCO_PREFIX}/lib/libmujoco.so")

if(NOT EXISTS "${mujoco_LIBRARY}")
    message(FATAL_ERROR "mujoco library not found at ${mujoco_LIBRARY}")
endif()

if(NOT TARGET mujoco::mujoco)
    add_library(mujoco::mujoco SHARED IMPORTED)
    set_target_properties(mujoco::mujoco PROPERTIES
        IMPORTED_LOCATION "${mujoco_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${mujoco_INCLUDE_DIR}"
    )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(mujoco DEFAULT_MSG mujoco_LIBRARY mujoco_INCLUDE_DIR)
