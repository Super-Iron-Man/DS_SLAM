cmake_minimum_required(VERSION 2.8)
project(DBoW2)

set(HDRS_DBOW2
    DBoW/BowVector.h
    DBoW/FORB.h
    DBow/FBrief.h
    DBoW/FClass.h
    DBoW/FeatureVector.h
    DBoW/ScoringObject.h
    DBoW/TemplatedVocabulary.h)

set(SRCS_DBOW2
    DBoW/BowVector.cpp
    DBoW/FORB.cpp
    DBoW/FBrief.cpp
    DBoW/FeatureVector.cpp
    DBoW/ScoringObject.cpp)

set(HDRS_DUTILS
    DUtils/Random.h
    DUtils/Timestamp.h
    DUtils/DException.h
    DUtils/DUtils.h)

set(SRCS_DUTILS
    DUtils/Random.cpp
    DUtils/Timestamp.cpp)

set(HDRS_DVISION
    DVision/BRIEF.h
    DVision/DVision.h
)

set(SRCS_DVISION
    DVision/BRIEF.cpp
)

set(SRCS_VINS_BIN
    VINS_bin/VocabularyBinary.cpp
)


find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
    find_package(OpenCV 2.4.3 QUIET)
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif()
endif()

#set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories(${OpenCV_INCLUDE_DIRS})

add_library(DBoW2 SHARED ${SRCS_DBOW2} ${SRCS_DUTILS} ${SRCS_DVISION} ${SRCS_VINS_BIN})

target_link_libraries(DBoW2 ${OpenCV_LIBS})