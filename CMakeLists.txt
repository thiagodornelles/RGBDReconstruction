cmake_minimum_required(VERSION 2.8)

project(DepthRGBD)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -W")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -msse4.2")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -msse4.2")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=c++11-narrowing")

set(project_sources
    main.cpp
    util.h
    fastbilateral.h
    filereader.h
    aligner.h
    pointcloudextend.h
    visibility.h
    scalableTSDF.h
)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# Find OpenMP
find_package(OpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(OPENMP_FOUND)

# Find Open3D
find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/lib/CMake)
link_directories(${Open3D_LIBRARY_DIRS})
include_directories(${Open3D_INCLUDE_DIRS})
list(APPEND Open3D_LIBRARIES dl)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Open3D_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Open3D_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${Open3D_EXE_LINKER_FLAGS}")


# Set OS-specific things here
#add_definitions(-DUNIX)
#add_compile_options(-Wno-deprecated-declarations)
#add_compile_options(-Wno-unused-result)
#add_definitions(-O3)

add_executable(${PROJECT_NAME} ${project_sources})
target_link_libraries(${PROJECT_NAME} ${Open3D_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} )
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
