rock_library(motion_planning
    SOURCES Dummy.cpp
    HEADERS Dummy.hpp)

rock_executable(motion_planning_bin Main.cpp
    DEPS motion_planning)

