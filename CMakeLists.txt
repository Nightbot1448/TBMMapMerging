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


# add_executable(load_state src/map_utils/load_state.cpp)
# add_executable(map_translation src/map_utils/map_translation.cpp)
# add_executable(parts_of_map src/map_utils/parts_of_map.cpp)
add_executable(conj_and_disj src/map_utils/conj_and_disj.cpp)
add_executable(merge_maps src/map_utils/merge_maps.cpp)
add_executable(comparator src/map_utils/comparator.cpp)


# target_link_libraries(load_state ${catkin_LIBRARIES} ${OpenCV_LIBS})
# target_link_libraries(map_translation ${catkin_LIBRARIES} ${OpenCV_LIBS})
# target_link_libraries(parts_of_map ${catkin_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(conj_and_disj ${catkin_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(merge_maps ${catkin_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(comparator ${catkin_LIBRARIES} ${OpenCV_LIBS})

install(
  TARGETS
  merge_maps
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(
  DIRECTORY config launch rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)