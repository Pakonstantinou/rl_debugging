#ifndef UNTITLED7_CRITIC_NETWORK_H
#define UNTITLED7_CRITIC_NETWORK_H
#include <torch/torch.h>

class criticgeneral : public torch::nn::Module {
public:
    virtual torch::Tensor forward(torch::Tensor states, torch::Tensor actions) = 0;
};

#endif // UNTITLED7_CRITIC_NETWORK_H
