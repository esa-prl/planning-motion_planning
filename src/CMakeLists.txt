include_directories(/usr/include/python3.8)
link_directories(/usr/lib/python3.8/config-3.8-x86_64-linux-gnu)

rock_library(motion_planning
    SOURCES MotionPlanning.cpp
    HEADERS MotionPlanning.hpp
    DEPS_PKGCONFIG base-types)

target_link_libraries(motion_planning python3.8)
