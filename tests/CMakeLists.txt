# ---------------------------------------------------------------------------
# Test executables and CTest registration
# All targets link against the `input_params` OBJECT library defined in the
# parent CMakeLists.txt, which already propagates MPI, HDF5, and yaml-cpp.
# ---------------------------------------------------------------------------

# --- yaml_test --------------------------------------------------------------
add_executable(test_input_params
    yaml_test/test_input_params.cpp
)
target_link_libraries(test_input_params PRIVATE input_params)

add_executable(test_input_params_mpi
    yaml_test/test_input_params_mpi.cpp
)
target_link_libraries(test_input_params_mpi PRIVATE input_params)

# --- logger_test ------------------------------------------------------------
add_executable(test_logger
    logger_test/test_logger.cpp
)
target_link_libraries(test_logger PRIVATE input_params)

# --- csv_test ---------------------------------------------------------------
add_executable(test_csv
    csv_test/test_csv.cpp
)
target_include_directories(test_csv PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_features(test_csv PRIVATE cxx_std_20)

# --- eigen_test -------------------------------------------------------------
add_executable(test_eigen_mpi
    eigen_test/test_eigen_mpi.cpp
)
target_link_libraries(test_eigen_mpi PRIVATE input_params Eigen3::Eigen)

# --- hdf5_test --------------------------------------------------------------
add_executable(test_h5io
    hdf5_test/test_h5io.cpp
)
target_link_libraries(test_h5io PRIVATE input_params Eigen3::Eigen ${MY_HDF5_LIBS})

# --- eikonal_test -----------------------------------------------------------
add_executable(test_eikonal
    eikonal_test/test_eikonal.cpp
)
target_link_libraries(test_eikonal PRIVATE input_params Eigen3::Eigen)
target_compile_options(test_eikonal PRIVATE -O2)

# ---------------------------------------------------------------------------
# CTest registration
# ---------------------------------------------------------------------------
add_test(
    NAME test_input_params
    COMMAND test_input_params ${CMAKE_SOURCE_DIR}/tests/yaml_test/input_params.yml
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
add_test(
    NAME test_input_params_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
            test_input_params_mpi ${CMAKE_SOURCE_DIR}/tests/yaml_test/input_params.yml
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)
add_test(
    NAME test_logger
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            test_logger
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
add_test(
    NAME test_csv
    COMMAND test_csv
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
add_test(
    NAME test_eigen_mpi
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
            test_eigen_mpi
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
add_test(
    NAME test_h5io
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
            test_h5io
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
add_test(
    NAME test_eikonal
    COMMAND test_eikonal
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
