#ifndef UNTITLED7_ACTOR_CRITIC_H
#define UNTITLED7_ACTOR_CRITIC_H
#include "algorithm.h"

class ac : public algorithm {
public:
private:
    //    std::shared_ptr<policygeneral> policy;
    //    std::shared_ptr<environment> env;
    //    std::shared_ptr<criticgeneral> critic;

public:
    ac(std::shared_ptr<policygeneral> pol, std::shared_ptr<criticgeneral> cr, std::shared_ptr<environment> envi, double policy_learning_rate, double critic_learning_rate) : algorithm(pol, cr, envi, policy_learning_rate, critic_learning_rate)
    {
    }

    void preprocess_states(torch::Tensor states)
    {
        curr_states = states.index({torch::indexing::Slice(0, states.size(0) - 1, 1)}).detach();
        next_states = states.index({torch::indexing::Slice(1, states.size(0), 1)}).detach();
    }

    void step()
    {
        torch::Tensor sigma1;
        torch::Tensor mu1;
        torch::Tensor logsigma1;
        std::tie(mu1, logsigma1) = policy->forward(curr_states);
        sigma1 = torch::exp(logsigma1);
        auto sampler1 = torch::randn({curr_states.size(0), 1}) * sigma1 + mu1;
        auto pdf = (1.0 / (sigma1 * std::sqrt(2.0 * M_PI))) * torch::exp(-0.5 * torch::pow((sampler1 - mu1) / sigma1, 2));
        auto log_prob = torch::log(pdf);

        torch::Tensor current_values = critic->forward(curr_states, actions);
        torch::Tensor q_target;
        {
            torch::NoGradGuard no_grad;

            torch::Tensor mu2;
            torch::Tensor logsigma2;
            std::tie(mu2, logsigma2) = policy->forward(next_states);
            torch::Tensor sigma2 = torch::exp(logsigma2);

            auto sampler2 = torch::randn({next_states.size(0), 1}) * sigma2 + mu2;
            torch::Tensor next_actions = sampler2; // to(torch::kDouble)
            next_actions = torch::squeeze(next_actions, 1);

            torch::Tensor next_values = critic->forward(next_states, next_actions);

            q_target = rewards.detach().view({-1, 1}) + dones.view({-1, 1}) * gamma * next_values - current_values;
        }

        torch::Tensor policy_loss_ = -torch::mean(current_values.detach() * log_prob);

        optimizer_policy->zero_grad();
        policy_loss_.backward();
        torch::nn::utils::clip_grad_value_(policy->parameters(), 5);
        optimizer_policy->step();

        torch::Tensor critic_loss_ = torch::nn::functional::mse_loss(current_values, q_target.detach(), torch::kMean);
        optimizer_critic->zero_grad();
        critic_loss_.backward();
        optimizer_critic->step();
    }
};
#endif // UNTITLED7_ACTOR_CRITIC_H
