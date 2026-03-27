cmake_minimum_required(VERSION 3.18)

set(CMAKE_PROJECT_NAME "SurfATT")
project( ${CMAKE_PROJECT_NAME} VERSION 2.0.0 LANGUAGES CXX Fortran)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set(CMAKE_CXX_FLAGS  "-g -Wall -DEIGEN_DONT_PARALLELIZE")
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU") 
  set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -march=native -O3")
  set(CMAKE_Fortran_FLAGS  "${CMAKE_Fortran_FLAGS} -march=native -ffixed-line-length-none")
  #set(CMAKE_CXX_FLAGS_RELEASE  "-O0 -D_GLIBCXX_DEBUG")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -xHost -DEIGEN_STRONG_INLINE=inline -O3")
  set(CMAKE_Fortran_FLAGS  "${CMAKE_Fortran_FLAGS} -xHost -extend-source")
else ()
  set(CMAKE_Fortran_FLAGS  "${CMAKE_Fortran_FLAGS} -march=native -ffixed-line-length-none")
  set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -march=native -O3")
  #set(CMAKE_CXX_FLAGS_RELEASE  "-O0 -D_GLIBCXX_DEBUG")
endif()

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
option(USE_EXTERNAL_LIBS OFF "Use system-installed libraries instead of bundled submodules")

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
if (yaml-cpp_FOUND AND NOT ${USE_EXTERNAL_LIBS})
    message(STATUS "Found yaml-cpp: ${YAML_CPP_INCLUDE_DIR}")
    # Normalize to yaml-cpp::yaml-cpp to avoid deprecated target warnings.
    if (NOT TARGET yaml-cpp::yaml-cpp)
        if (TARGET yaml-cpp)
            add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
        else()
            add_library(yaml-cpp::yaml-cpp INTERFACE IMPORTED)
            if (YAML_CPP_INCLUDE_DIR)
                set_property(TARGET yaml-cpp::yaml-cpp APPEND PROPERTY
                    INTERFACE_INCLUDE_DIRECTORIES "${YAML_CPP_INCLUDE_DIR}")
            endif()
            if (YAML_CPP_LIBRARIES)
                set_property(TARGET yaml-cpp::yaml-cpp APPEND PROPERTY
                    INTERFACE_LINK_LIBRARIES "${YAML_CPP_LIBRARIES}")
            endif()
        endif()
    endif()
else()
    message(STATUS "yaml-cpp not found via find_package, using submodule")
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    add_subdirectory(external_libs/yaml-cpp)
    if (TARGET yaml-cpp AND NOT TARGET yaml-cpp::yaml-cpp)
        add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
    endif()
endif()

# ------------------------------------------------------
# find spdlog
find_package(spdlog QUIET)
if (spdlog_FOUND AND NOT ${USE_EXTERNAL_LIBS})
    message(STATUS "spdlog found")
else()
  message(STATUS "spdlog not found. use source code external_libs/spdlog...")
  initialize_submodule(external_libs/spdlog)
  add_subdirectory(external_libs/spdlog EXCLUDE_FROM_ALL)
endif()

if (TARGET spdlog::spdlog_header_only)
    set(SPDLOG_TARGET spdlog::spdlog_header_only)
elseif(TARGET spdlog::spdlog)
    set(SPDLOG_TARGET spdlog::spdlog)
elseif(TARGET spdlog)
    add_library(spdlog::spdlog ALIAS spdlog)
    set(SPDLOG_TARGET spdlog::spdlog)
else()
    message(FATAL_ERROR "spdlog target not found after discovery")
endif()
message(STATUS "Using spdlog target: ${SPDLOG_TARGET}")

# ---------------------------------------------------------------------------
# Eigen3
# ---------------------------------------------------------------------------
find_package(Eigen3 REQUIRED)
if (Eigen3_FOUND)
    message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Eigen3 not found")
endif()
# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

add_subdirectory("src/surfker")

file(GLOB SOURCES "src/*.cpp")
add_library(objs OBJECT ${SOURCES} ${SURFKER_SOURCES})
target_include_directories(objs PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(objs PUBLIC
    Eigen3::Eigen
    yaml-cpp::yaml-cpp
    ${SPDLOG_TARGET}
    MPI::MPI_CXX
    ${MY_HDF5_LIBS}
    surfker_obj
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()
add_subdirectory(tests)
