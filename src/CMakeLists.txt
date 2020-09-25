include_directories(/usr/include/python3.6m)
link_directories(/usr/lib/python3.6/config-3.6m-x86_64-linux-gnu)

rock_library(motion_planning
    SOURCES MotionPlanning.cpp
    HEADERS MotionPlanning.hpp
	DEPS_PKGCONFIG base-types)

target_link_libraries(motion_planning python3.6m)
