CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

SET(PROJECT_NAME ds_slam)
PROJECT(${PROJECT_NAME})

#set(CMAKE_VERBOSE_MAKEFILE ON)

#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
#set(CMAKE_BUILD_TYPE Release)

set(EXECUTABLE_OUTPUT_PATH bin)
set(LIBRARY_OUTPUT_PATH lib)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# required libraries
find_package(SuiteParse REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS system thread)
find_package(Ceres REQUIRED)

# optional libraries
find_package(LibZip QUIET)
find_package(Pangolin 0.2 QUIET)
#find_package(OpenCV QUIET)
find_package(OpenCV REQUIRED)

# flags
add_definitions("-DENABLE_SSE")
if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    message("debug mode!")
    set(CMAKE_CXX_FLAGS
        "${SSE_FLAGS} -O0 -g -std=c++11 -march=native -Wall")
elseif(${CMAKE_BUILD_TYPE} MATCHES "RelWithDebInfo")
    message("release with debug info mode!")
    set(CMAKE_CXX_FLAGS
        "${SSE_FLAGS} -O3 -g -std=c++11 -march=native -Wall")
else()
    message("release mode!")
    set(CMAKE_CXX_FLAGS
        "${SSE_FLAGS} -O3 -std=c++11 -march=native -Wall")
endif()

# 3-th party libs
add_subdirectory(Thirdparty/DBoW2)

# Sources files
set(DS_SLAM_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/Core/FullSystem/full_system.cpp
    ${PROJECT_SOURCE_DIR}/Core/FullSystem/full_system_optimize.cpp
    ${PROJECT_SOURCE_DIR}/Core/FullSystem/full_system_optimize_point.cpp
    ${PROJECT_SOURCE_DIR}/Core/FullSystem/full_system_marginalize.cpp
    ${PROJECT_SOURCE_DIR}/Core/FullSystem/full_system_debug_stuff.cpp
    ${PROJECT_SOURCE_DIR}/Core/Common/residuals.cpp
    ${PROJECT_SOURCE_DIR}/Core/Common/immature_point.cpp
    ${PROJECT_SOURCE_DIR}/Core/Common/hessian_blocks.cpp
    ${PROJECT_SOURCE_DIR}/Core/Common/pixel_selector.cpp
    ${PROJECT_SOURCE_DIR}/Core/Common/frame.cpp
    ${PROJECT_SOURCE_DIR}/Core/Common/point.cpp
    ${PROJECT_SOURCE_DIR}/Core/Tracker/coarse_tracker.cpp
    ${PROJECT_SOURCE_DIR}/Core/Initializer/coarse_initializer.cpp
    ${PROJECT_SOURCE_DIR}/Core/Optimization/energy_function.cpp
    ${PROJECT_SOURCE_DIR}/Core/Optimization/accumulated_top_hessian.cpp
    ${PROJECT_SOURCE_DIR}/Core/Optimization/accumulated_SC_hessian.cpp
    ${PROJECT_SOURCE_DIR}/Core/Optimization/energy_function_structs.cpp
    ${PROJECT_SOURCE_DIR}/Core/LoopClosing/brief_extractor.cpp
    ${PROJECT_SOURCE_DIR}/Core/LoopClosing/orb_extractor.cpp
    ${PROJECT_SOURCE_DIR}/Core/LoopClosing/keyframe.cpp
    ${PROJECT_SOURCE_DIR}/Core/LoopClosing/pose_graph.cpp
    ${PROJECT_SOURCE_DIR}/Core/Utils/settings.cpp
    ${PROJECT_SOURCE_DIR}/Core/Utils/undistort.cpp
    ${PROJECT_SOURCE_DIR}/Core/Utils/global_calib.cpp
    ${PROJECT_SOURCE_DIR}/Core/Utils/utility.cpp
)

# include files
include_directories(
    ${PROJECT_SOURCE_DIR}/Core
    ${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
    ${PROJECT_SOURCE_DIR}/Thirdparty/sse2neon
    ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2
    ${EIGEN3_INCLUDE_DIR}
    ${CERES_INCLUDE_DIRS}
)

# decide if we have pangolin
if (Pangolin_FOUND)
    message("--- found PANGOLIN, compiling dso_pangolin library.")
    include_directories( ${Pangolin_INCLUDE_DIRS} ) 
    set(DISPLAY_PANGOLIN_SOURCE_FILES 
      ${PROJECT_SOURCE_DIR}/Core/Visualizer/Pangolin/keyframe_display.cpp
      ${PROJECT_SOURCE_DIR}/Core/Visualizer/Pangolin/visualizer_pangolin.cpp)
    set(HAS_PANGOLIN 1)
else ()
    message("--- could not find PANGOLIN, not compiling dso_pangolin library.")
    message("    this means there will be no 3D display / GUI available for dso_dataset.")
    set(DISPLAY_PANGOLIN_SOURCE_FILES )
    set(HAS_PANGOLIN 0)
endif ()

# we have openCV
message("--- found OpenCV, compiling dso_opencv library.")
include_directories( ${OpenCV_INCLUDE_DIRS} )
set(DISPLAY_OPENCV_SOURCE_FILES 
  ${PROJECT_SOURCE_DIR}/Core/Visualizer/image_display_OpenCV.cpp
  ${PROJECT_SOURCE_DIR}/Core/Visualizer/image_RW_OpenCV.cpp)
set(HAS_OPENCV 1)


# decide if we have ziplib.
if (LIBZIP_LIBRARY)
    message("--- found ziplib (${LIBZIP_VERSION}), compiling with zip capability.")
    add_definitions(-DHAS_ZIPLIB=1)
    include_directories( ${LIBZIP_INCLUDE_DIR_ZIP} ${LIBZIP_INCLUDE_DIR_ZIPCONF} ) 
else()
    message("--- not found ziplib (${LIBZIP_LIBRARY}), compiling without zip capability.")
    set(LIBZIP_LIBRARY "")
endif()


# compile main library.
include_directories(${CSPARSE_INCLUDE_DIR} ${CHOLMOD_INCLUDE_DIR}) 
add_library(ds_slam ${DS_SLAM_SOURCE_FILES} 
    ${DISPLAY_OPENCV_SOURCE_FILES} 
    ${DISPLAY_PANGOLIN_SOURCE_FILES}
)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin") # OSX
    set(BOOST_THREAD_LIBRARY boost_thread-mt)
else()
    set(BOOST_THREAD_LIBRARY boost_thread)
endif()

# build main executable (only if we have both OpenCV and Pangolin)
if (Pangolin_FOUND)
    message("--- compiling DS_SLAM_dataset.")
    add_executable(ds_slam_dataset ${PROJECT_SOURCE_DIR}/Examples/main_dataset.cpp )
    target_link_libraries(ds_slam_dataset ds_slam 
        boost_system 
        cxsparse 
        ${BOOST_THREAD_LIBRARY} 
        ${LIBZIP_LIBRARY} 
        ${Pangolin_LIBRARIES} 
        ${OpenCV_LIBS}
        ${CERES_LIBRARIES}
        DBoW2
    )
else()
    message("--- not building DS_SLAM_dataset, since either don't have openCV or Pangolin.")
endif()

