#ifndef FRAMEWORK_DPG_H
#define FRAMEWORK_DPG_H
#include "algorithm.h"

 class dpg : public algorithm {

 private:
//    std::shared_ptr<policygeneral> policy;
//    std::shared_ptr<environment> env;
//    std::shared_ptr<criticgeneral> critic;

public:

    dpg(std::shared_ptr<policygeneral> pol, std::shared_ptr<criticgeneral> cr, std::shared_ptr<environment> envi,
        double policy_learning_rate, double critic_learning_rate) : algorithm(pol, cr, envi, policy_learning_rate,
                                                                              critic_learning_rate) {
    }

//    deterministic policy;
//    criticgeneral critic;
//    torch::optim::Adam optimizer_policy(policy->parameters(), /*lr=*/0.0002);
//    torch::optim::Adam optimizer_critic(critic->parameters(), /*lr=*/0.001);
//    policy->to(torch::kDouble);
//    critic->to(torch::kDouble);


    void preprocess_states(torch::Tensor states) {
        curr_states = states.index({torch::indexing::Slice(0, states.size(0) - 1, 1)}).detach();
        next_states = states.index({torch::indexing::Slice(1, states.size(0), 1)}).detach();

    }

    void step() {

        torch::Tensor current_values = critic->forward(curr_states, actions);

        torch::Tensor q_target;
        {
            torch::NoGradGuard no_grad;
            torch::Tensor next_actions;
            torch::Tensor empty;
            std::tie(next_actions, empty) = policy->forward(next_states);
            next_actions = torch::squeeze(next_actions, 1);
            torch::Tensor next_values = critic->forward(next_states, next_actions);
            q_target = rewards.detach().view({-1, 1}) + dones.view({-1, 1}) * gamma * next_values - current_values;
        }


        torch::Tensor acts;
        torch::Tensor empty;
        std::tie(acts, empty) = policy->forward(curr_states);
        acts = torch::squeeze(acts, 1);

        torch::Tensor cv = critic->forward(curr_states, acts);

        auto policy_loss = -torch::mean(cv);
        auto critic_loss = torch::nn::functional::mse_loss(current_values, q_target.detach(), torch::kMean);

        optimizer_policy->zero_grad();
        policy_loss.backward();
        optimizer_policy->step();

        optimizer_critic->zero_grad();
        critic_loss.backward();
        optimizer_critic->step();
    }
};
#endif //FRAMEWORK_DPG_H
