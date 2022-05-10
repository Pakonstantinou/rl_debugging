#ifndef UNTITLED7_ENVIRONMENT_H
#define UNTITLED7_ENVIRONMENT_H

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <robot_dart/control/simple_control.hpp>
#include <robot_dart/robot_dart_simu.hpp>
#include <string>
#include <torch/torch.h>
#include <tuple>

class environment {
public:
    virtual torch::Tensor reset() = 0;

    virtual std::tuple<torch::Tensor, double, bool, bool> step(torch::Tensor move) = 0;
};

#endif // UNTITLED7_ENVIRONMENT_H
