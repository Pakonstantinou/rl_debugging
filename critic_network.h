#ifndef rl_debugging_CRITIC_NETWORK_H
#define rl_debugging_CRITIC_NETWORK_H
#include <torch/torch.h>

class criticgeneral : public torch::nn::Module {
public:
    virtual torch::Tensor forward(torch::Tensor states, torch::Tensor actions) = 0;
};

#endif // rl_debugging_CRITIC_NETWORK_H
