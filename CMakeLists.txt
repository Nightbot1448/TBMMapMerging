cmake_minimum_required(VERSION 2.8.12)
project(map_merging)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -pthread -fdiagnostics-color")


if(${CMAKE_VERSION} VERSION_LESS "3.1")
  add_compile_options(-std=c++14)
else()
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()


find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  roslz4
)
  
find_package( OpenCV REQUIRED )

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_package()


add_executable(load_state src/map_utils/load_state.cpp)
add_executable(comparator src/map_utils/comparator.cpp)
add_executable(map_merging src/map_merging/map_merging.cpp)


target_link_libraries(load_state ${catkin_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(comparator ${catkin_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(map_merging ${catkin_LIBRARIES} ${OpenCV_LIBS})
