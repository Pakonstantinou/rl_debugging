cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(rl_debugging)
set(CMAKE_PREFIX_PATH "/home/pakos/libtorch/share/cmake/Torch")
# we ask for Magnum because we want to build the graphics
find_package(RobotDART REQUIRED OPTIONAL_COMPONENTS Magnum)
find_package(Boost REQUIRED COMPONENTS filesystem)
find_package (Threads)
find_package(Torch REQUIRED)
add_executable(robot_dart environment.h pendulum.h algorithm.h main.cpp policy.h dpg.h actor_critic.h reinforce.h critic_network.h)

set_property(TARGET robot_dart PROPERTY CXX_STANDARD 11)

target_link_libraries(robot_dart
        ${CMAKE_THREAD_LIBS_INIT}
        ${TORCH_LIBRARIES}
        RobotDART::Simu
        Boost::filesystem
        )

set_property(TARGET robot_dart PROPERTY CXX_STANDARD 11)


if(RobotDART_Magnum_FOUND)
    add_executable(robot_dart_graphics environment.h pendulum.h algorithm.h main.cpp policy.h dpg.h actor_critic.h reinforce.h critic_network.h)
    target_link_libraries(robot_dart_graphics
            ${CMAKE_THREAD_LIBS_INIT}
            RobotDART::Simu
            RobotDART::Magnum
            Boost::filesystem
            "${TORCH_LIBRARIES}"
            #${PYTHON_LIBRARIES}
            )
    set_property(TARGET robot_dart_graphics PROPERTY CXX_STANDARD 11)

endif()
