#ifndef FRAMEWORK_ALGORITHM_H
#define FRAMEWORK_ALGORITHM_H
#include "critic_network.h"
#include "environment.h"
#include "pendulum.h"
#include "policy.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <torch/torch.h>

class algorithm {

private:
public:
    std::shared_ptr<policygeneral> policy;
    std::shared_ptr<environment> env;
    std::shared_ptr<criticgeneral> critic;
    std::shared_ptr<torch::optim::Adam> optimizer_policy;
    std::shared_ptr<torch::optim::Adam> optimizer_critic;

    algorithm(std::shared_ptr<policygeneral> pol, std::shared_ptr<criticgeneral> cr, std::shared_ptr<environment> envi, double policy_learning_rate, double critic_learning_rate)
    {

        policy = pol;
        env = envi;
        critic = cr;
        optimizer_policy = std::make_shared<torch::optim::Adam>(policy->parameters(), /*lr=*/policy_learning_rate);
        optimizer_critic = std::make_shared<torch::optim::Adam>(critic->parameters(), /*lr=*/critic_learning_rate);
    }

    torch::Tensor returns;
    torch::Tensor curr_states;
    torch::Tensor next_states;
    torch::Tensor actions;
    torch::Tensor rewards;
    torch::Tensor dones;

    torch::Tensor states;
    double gamma = 0.999;
    double sum_of_rewards;

    void run_episode()
    {
        double reward;
        bool done;
        bool success = false;
        sum_of_rewards = 0;
        torch::Tensor state = env->reset(); // correct?
        states = env->reset(); // correct?
        states = states.unsqueeze(0).clone();
        torch::Tensor tempreward;
        int index = 0;
        std::vector<int> num_dones;
        std::vector<double> gt;
        double temp_gt;

        // zero these
        // correct??
        actions = torch::empty({0});
        rewards = torch::empty({0});
        dones = torch::empty({0});
        curr_states = torch::empty({0});
        next_states = torch::empty({0});
        returns = torch::empty({0});

        {
            torch::NoGradGuard no_grad;
            while (true) {
                // calculate probabilities of taking each action
                torch::Tensor action = policy->act(state);

                done = false;

                // use action in the environment
                std::tie(state, reward, done, success) = env->step(action);


                if (states.size(0) == 0)
                    states = state.unsqueeze(0);
                else
                    states = torch::cat({states, state.unsqueeze(0)}, 0);

                if (actions.size(0) == 0)
                    actions = action;
                else
                    actions = torch::cat({actions, action});

                if (rewards.size(0) == 0)
                    rewards = torch::tensor({reward});
                else {
                    tempreward = torch::tensor({reward});
                    rewards = torch::cat({rewards, tempreward});
                }

                if (done == 0) {
                    num_dones.push_back(1);
                }
                else {
                    num_dones.push_back(0);
                }

                if (done) {
                    env->reset();
                    break;
                }
            }

            int numdonessize = num_dones.size();
            dones = torch::from_blob(num_dones.data(), {numdonessize}, torch::kInt32).to(torch::kDouble);
            torch::Tensor a = torch::sum(rewards);
            sum_of_rewards = a.item<double>();

            for (int i = 0; i < rewards.size(0); i++) {
                temp_gt = temp_gt + rewards[i].item<double>() * pow(gamma, i);
                gt.push_back(temp_gt);
            }

            returns = torch::from_blob(gt.data(), {rewards.size(0)});
            //std::cout<<gt<<std::endl<<returns<<std::endl;
        }
    }

    virtual void preprocess_states(torch::Tensor states) = 0;

    virtual void step() = 0;
};
#endif