#ifndef UNTITLED7_REINFORCE_H
#define UNTITLED7_REINFORCE_H
#include "algorithm.h"

class reinforce : public algorithm {

private:
    //    std::shared_ptr<policygeneral> policy;
    //    std::shared_ptr<environment> env;
    //    std::shared_ptr<criticgeneral> critic;

public:
    reinforce(std::shared_ptr<policygeneral> pol, std::shared_ptr<criticgeneral> cr, std::shared_ptr<environment> envi, double policy_learning_rate, double critic_learning_rate) : algorithm(pol, cr, envi, policy_learning_rate, critic_learning_rate)
    {
    }

    void preprocess_states(torch::Tensor states)
    {
    }

    void step()
    {
        torch::Tensor sigma2;
        torch::Tensor mu2;
        torch::Tensor logsigma2;
        std::tie(mu2, logsigma2) = policy->forward(states);
        sigma2 = torch::exp(logsigma2);
        auto sampler2 = torch::randn({states.size(0), 1}) * sigma2 + mu2;
        auto pdf2 = (1.0 / (sigma2 * std::sqrt(2 * M_PI))) * torch::exp(-0.5 * torch::pow((sampler2 - mu2) / sigma2, 2));
        auto log_prob2 = torch::log(pdf2);
        torch::Tensor policy_loss_ = torch::sum(log_prob2 * returns);

        optimizer_policy->zero_grad();
        policy_loss_.backward();
        optimizer_policy->step();
    }
};
#endif // UNTITLED7_REINFORCE_H
