#ifndef rl_debugging_PENDULUM_H
#define rl_debugging_PENDULUM_H

#include "environment.h"

/*#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;*/

#ifdef GRAPHIC
#include <robot_dart/gui/magnum/graphics.hpp>
#endif

namespace rd = robot_dart;

class pendulum : public environment {

public:
    torch::Tensor _state;
    // bool done;
    // double reward;
    int simusteps = 0;
    std::shared_ptr<robot_dart::Robot> robot = std::make_shared<robot_dart::Robot>("pendulum.urdf");
    std::shared_ptr<robot_dart::RobotDARTSimu> simu = std::make_shared<robot_dart::RobotDARTSimu>(0.001);

    pendulum()
    {
        robot->fix_to_world();
        robot->set_position_enforced(true);
        robot->set_actuator_types("torque");
        simu->add_robot(robot);
#ifdef GRAPHIC
        auto graphics = std::make_shared<robot_dart::gui::magnum::Graphics>();
        simu->set_graphics(graphics);
#endif
        //_state = torch::tensor({M_PI}, torch::kDouble);

        reset();
    }

    std::tuple<torch::Tensor, double, bool, bool> step(torch::Tensor act) override
    {
        bool success = false;
        double move = act.item<double>();

        auto cmds = rd::make_vector({move});

        for (int i = 0; i < 50; i++) {
            robot->set_commands(cmds);
            simu->step_world();
            simusteps++;
        }

        double temp_pos = robot->positions()[0];
        // double data[] = {temp_pos};
        double sin_pos = std::sin(temp_pos);
        double cos_pos = std::cos(temp_pos);
        bool done = false;
        double reward;
        double temp_velocity = robot->velocities()[0];
        double theta = angle_dist(temp_pos, 0.);
        // reward=-(std::abs(M_PI-robot->positions()[0]));
        reward = -theta;
        // reward = -0.01 * move;
        // reward = -std::abs(temp_velocity);

        // if (std::abs(M_PI-temp_pos)<0.0001) {
        if (theta < 0.1) {
            // if ((std::abs(theta)<0.1)){

            // auto cmds = rd::make_vector({0});
            // robot->set_commands(cmds);
            done = false;
            reward = 10;
            // simusteps = 0;
            // std::cout << "success" << std::endl;
            success = true;
            // std::cout<<temp_pos<<std::endl;

            // torch::Tensor reset();
        }
        if (simusteps == 5000) {
            // auto cmds = rd::make_vector({0});
            // robot->set_commands(cmds);
            done = true;
            simusteps = 0;
            // std::cout << "fail" << std::endl;
            success = false;
            // std::cout << temp_pos << std::endl;
            // torch::Tensor reset();
        }

        //_state = torch::from_blob(data, {1}, torch::TensorOptions().dtype(torch::kDouble));
        _state = torch::tensor({sin_pos, cos_pos, temp_velocity}, torch::kDouble);

        return {_state, reward, done, success};
    }

    torch::Tensor reset() override
    {
        simusteps = 0;
        robot->reset();
        robot->set_positions(robot_dart::make_vector({0}));
        double temp_pos = robot->positions()[0];
        double sin_pos = std::sin(temp_pos);
        double cos_pos = std::cos(temp_pos);
        _state = torch::tensor({sin_pos, cos_pos, 0.0}, torch::kDouble);
        return _state;
    }

    double angle_dist(double a, double b)
    {
        double theta = b - a;
        while (theta < -M_PI)
            theta += 2 * M_PI;
        while (theta > M_PI)
            theta -= 2 * M_PI;
        return std::abs(theta);
    }
};

#endif // rl_debugging_PENDULUM_H
