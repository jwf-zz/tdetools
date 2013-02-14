cmake_minimum_required(VERSION 2.8)

project(tdetools)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(OpenCV REQUIRED)

find_package(ANN REQUIRED)

add_subdirectory(external/Tisean)
include_directories(external)

add_executable(buildtree BuildTree.cpp TDEModel.cpp Utils.cpp)
target_link_libraries(buildtree ddtsa ${OpenCV_LIBS} ${ANN_LIBRARIES}) 

add_executable(classifytrajectory ClassifyTrajectory.cpp Classifier.cpp TDEModel.cpp Utils.cpp)
target_link_libraries(classifytrajectory ddtsa ${OpenCV_LIBS} ${ANN_LIBRARIES}) 
