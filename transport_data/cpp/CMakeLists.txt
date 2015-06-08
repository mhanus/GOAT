cmake_minimum_required(VERSION 2.8)

set(PROJECT_NAME cpp)
project(${PROJECT_NAME})


# Set verbose output while testing CMake
#set(CMAKE_VERBOSE_MAKEFILE 1)

# Set CMake behavior
cmake_policy(SET CMP0004 OLD)

# Get DOLFIN configuration data (DOLFINConfig.cmake must be in DOLFIN_CMAKE_CONFIG_PATH)
find_package(DOLFIN)

# Compiler definitions
add_definitions(${DOLFIN_CXX_DEFINITIONS})

# Include directories
include_directories(${DOLFIN_INCLUDE_DIRS})
include_directories(SYSTEM ${DOLFIN_3RD_PARTY_INCLUDE_DIRS})

# Executable
add_executable(${PROJECT_NAME} main.cpp OrdinatesData.cpp AngularTensors.cpp)

# Target libraries
target_link_libraries(${PROJECT_NAME} ${DOLFIN_LIBRARIES} ${DOLFIN_3RD_PARTY_LIBRARIES})
