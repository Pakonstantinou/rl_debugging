#ifndef rl_debugging_POLICY_H
#define rl_debugging_POLICY_H
#include <torch/torch.h>

class policygeneral : public torch::nn::Module {

public:
    virtual std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) = 0;

    virtual torch::Tensor act(torch::Tensor state) = 0;
};

class stochastic : public policygeneral {
public:
    virtual std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) = 0; //////correct?

    virtual torch::Tensor act(torch::Tensor state)
    {
        torch::Tensor mu;
        torch::Tensor logsigma;

        std::tie(mu, logsigma) = forward(state); ///
        auto sample = torch::randn({1}, torch::kDouble) * torch::exp(logsigma) + mu;

        torch::Tensor action = (torch::tanh(sample).to(torch::kDouble))*10.0; ////////////////TESTING

        //torch::Tensor action = sample.to(torch::kDouble);

        return action;
    }
};

class deterministic : public policygeneral {
public:
    virtual std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) = 0;

    virtual torch::Tensor act(torch::Tensor state)
    {
        torch::Tensor action;
        torch::Tensor logsigma;
        std::tie(action, logsigma) = forward(state);
        auto noise = torch::randn({1}) * 0.2 + 0;
        action = action + noise;
        return action;
    }
};

#endif // rl_debugging_POLICY_H
