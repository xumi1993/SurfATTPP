file(GLOB MINPACK_SOURCES "*.f90" "*.cpp")
add_library(minpack_obj OBJECT ${MINPACK_SOURCES})
target_include_directories(minpack_obj PUBLIC ${CMAKE_SOURCE_DIR}/include)
