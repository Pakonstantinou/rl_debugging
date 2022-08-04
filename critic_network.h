#ifndef FRAMEWORK_CRITIC_NETWORK_H
#define FRAMEWORK_CRITIC_NETWORK_H
#include <torch/torch.h>

class criticgeneral : public torch::nn::Module {
public:
    virtual torch::Tensor forward(torch::Tensor states, torch::Tensor actions) = 0;
};

#endif // FRAMEWORK_CRITIC_NETWORK_H
