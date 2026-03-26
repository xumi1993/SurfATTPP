cmake_minimum_required(VERSION 3.18)

set(CMAKE_PROJECT_NAME "SurfATT")
project( ${CMAKE_PROJECT_NAME} VERSION 2.0.0 LANGUAGES CXX)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# ---------------------------------------------------------------------------
# Precision option  (-DUSE_SINGLE_PRECISION=ON to use float)
# ---------------------------------------------------------------------------
option(USE_SINGLE_PRECISION "Use single-precision (float) throughout" OFF)
if (USE_SINGLE_PRECISION)
    add_compile_definitions(USE_SINGLE_PRECISION)
    message(STATUS "Precision: float (single)")
else()
    message(STATUS "Precision: double (default)")
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

# MPI (required)
find_package(MPI REQUIRED)
message(STATUS "MPI_CXX_COMPILER: ${MPI_CXX_COMPILER}")
message(STATUS "MPI_CXX_INCLUDE_DIRS: ${MPI_CXX_INCLUDE_DIRS}")

# find HDF5 parallel
message(STATUS "Running CMAKE FindHDF5.cmake...")
# set parallel HDF5 default
find_package(HDF5 REQUIRED COMPONENTS C CXX CONFIG QUIET)
if (NOT HDF5_FOUND)
  message(STATUS "HDF5 not found with CONFIG mode, try without CONFIG mode...")
  find_package(HDF5 REQUIRED COMPONENTS C CXX)
endif()
if(HDF5_FOUND)
  if(TARGET hdf5::hdf5_cpp-shared AND TARGET hdf5::hdf5-shared)
      set(MY_HDF5_LIBS hdf5::hdf5_cpp-shared hdf5::hdf5-shared)
  elseif(TARGET hdf5::hdf5_cpp AND TARGET hdf5::hdf5)
      set(MY_HDF5_LIBS hdf5::hdf5_cpp hdf5::hdf5)
  elseif(TARGET hdf5::hdf5_cpp-static AND TARGET hdf5::hdf5-static)
      set(MY_HDF5_LIBS hdf5::hdf5_cpp-static hdf5::hdf5-static)
  elseif(TARGET hdf5-shared AND TARGET hdf5_cpp-shared)
      set(MY_HDF5_LIBS hdf5_cpp-shared hdf5-shared)
  elseif(TARGET hdf5 AND TARGET hdf5_cpp)
      set(MY_HDF5_LIBS hdf5_cpp hdf5)
  else()
    set(MY_HDF5_LIBS ${HDF5_LIBRARIES})
    if("${MY_HDF5_LIBS}" STREQUAL "")
        set(MY_HDF5_LIBS ${HDF5_C_LIBRARIES} ${HDF5_CXX_LIBRARIES})
    endif()
  endif()
  message(STATUS "HDF5_FOUND: ${HDF5_FOUND}")
  message(STATUS "MY_HDF5_LIBS: ${MY_HDF5_LIBS}")
  add_definitions(-DUSE_HDF5)
  include_directories(${HDF5_INCLUDE_DIRS})
endif()

# submodules
add_subdirectory(external_libs)

# yaml-cpp: prefer a system installation, fall back to the bundled submodule.
find_package(yaml-cpp QUIET)
if (yaml-cpp_FOUND)
    message(STATUS "Found yaml-cpp: ${YAML_CPP_INCLUDE_DIR}")
    # Some older cmake configs export the include path only as a variable, not
    # as a target property.  Create an INTERFACE target so downstream code can
    # always use target_link_libraries(... yaml-cpp).
    if (NOT TARGET yaml-cpp)
        add_library(yaml-cpp INTERFACE IMPORTED)
    endif()
    set_property(TARGET yaml-cpp APPEND PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES "${YAML_CPP_INCLUDE_DIR}")
    set_property(TARGET yaml-cpp APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES "${YAML_CPP_LIBRARIES}")
else()
    message(STATUS "yaml-cpp not found via find_package, using submodule")
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    add_subdirectory(external_libs/yaml-cpp)
endif()

# ------------------------------------------------------
# find spdlog
find_package(spdlog QUIET)
if (spdlog_FOUND AND NOT ${USE_EXTERNAL_LIBS})
  message(STATUS "spdlog found")
  message(STATUS "SPDLOG_INCLUDE_DIRS: ${SPDLOG_INCLUDE_DIRS}")
  message(STATUS "SPDLOG_LIBRARIES: ${SPDLOG_LIBRARIES}")
else()
  message(STATUS "spdlog not found. use source code external_libs/spdlog...")
  initialize_submodule(external_libs/spdlog)
  add_subdirectory(external_libs/spdlog EXCLUDE_FROM_ALL)
  set(SPDLOG_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/external_libs/spdlog/include)
endif()
include_directories(${SPDLOG_INCLUDE_DIRS})

# ---------------------------------------------------------------------------
# Eigen3
# ---------------------------------------------------------------------------
set(EIGEN3_HINT "/Users/xumijian/opt/eigen-5.0.0")
find_package(Eigen3 QUIET HINTS "${EIGEN3_HINT}/share/eigen3/cmake"
                                "${EIGEN3_HINT}/lib/cmake/eigen3"
                                "${EIGEN3_HINT}/share/cmake/Modules")
if (Eigen3_FOUND OR EIGEN3_FOUND)
    message(STATUS "Eigen3 found via find_package")
else()
    # Fall back to manual header-only include
    if (EXISTS "${EIGEN3_HINT}/include/eigen3/Eigen/Core")
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_HINT}/include/eigen3")
        message(STATUS "Eigen3 set up manually from ${EIGEN3_HINT}/include/eigen3")
    else()
        message(WARNING "Eigen3 not found — eigen tests will not build correctly")
    endif()
endif()

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

file(GLOB SOURCES "src/*.cpp")
add_library(input_params OBJECT ${SOURCES})
target_include_directories(input_params PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(input_params PUBLIC yaml-cpp MPI::MPI_CXX ${MY_HDF5_LIBS})

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
