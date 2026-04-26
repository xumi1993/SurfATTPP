file(GLOB SURFKER_SOURCES "*.f*")
file(GLOB SURFKER_C_SOURCES "*.cpp")
include_directories(${PROJECT_SOURCE_DIR}/include)
add_library(surfker_obj OBJECT ${SURFKER_SOURCES} ${SURFKER_C_SOURCES})
