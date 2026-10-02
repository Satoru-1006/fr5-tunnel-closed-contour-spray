cmake_minimum_required(VERSION 3.16)
project(p2b3_c5b0_bullet_fixture LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Bullet REQUIRED)
add_executable(p2b3_c5b0_bullet_fixture p2b3_c5b0_bullet_fixture.cpp)
target_include_directories(p2b3_c5b0_bullet_fixture PRIVATE ${BULLET_INCLUDE_DIRS})
target_link_libraries(p2b3_c5b0_bullet_fixture PRIVATE ${BULLET_LIBRARIES})
