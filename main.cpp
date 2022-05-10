#include "actor_critic.h"
#include "dpg.h"
#include "reinforce.h"

class MyPolicy : public stochastic {
public:
    torch::nn::Linear fc1, out, out_logsigma;

    MyPolicy() : fc1(register_module("fc1", torch::nn::Linear(3, 64))),
                 out(register_module("out", torch::nn::Linear(64, 1))),
                 out_logsigma(register_module("out_sigma", torch::nn::Linear(64, 1))) {}

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x)
    {
        x = torch::relu(fc1(x));
        torch::Tensor mu = out(x);
        torch::Tensor logsigma = out_logsigma(x);

        return {mu, logsigma};
    }
};

class MyPolicy2 : public deterministic {
public:
    torch::nn::Linear fc1, out, out_logsigma;

    MyPolicy2() : fc1(register_module("fc1", torch::nn::Linear(3, 64))),
                  out(register_module("out", torch::nn::Linear(64, 1))),
                  out_logsigma(register_module("out_sigma", torch::nn::Linear(64, 1))) {}

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x)
    {
        x = torch::relu(fc1(x));
        torch::Tensor mu = out(x);
        torch::Tensor logsigma = out_logsigma(x);

        return {mu, logsigma};
    }
};

class MyCritic : public criticgeneral {
public:
    torch::nn::Linear fc1, out;

    MyCritic() : fc1(register_module("fc1", torch::nn::Linear(4, 64))),
                 out(register_module("out", torch::nn::Linear(64, 1))) {}

    torch::Tensor forward(torch::Tensor states, torch::Tensor action)
    {
        torch::Tensor x = torch::cat({states, action.unsqueeze(1)}, 1); // check for errors
        x = torch::relu(fc1(x));
        x = out(x);

        return {x};
    }
};

int main()
{

    // example

    std::shared_ptr<policygeneral> pol = std::make_shared<MyPolicy>();
    std::shared_ptr<criticgeneral> cr = std::make_shared<MyCritic>();
    std::shared_ptr<environment> env = std::make_shared<pendulum>();

    std::shared_ptr<policygeneral> pol2 = std::make_shared<MyPolicy2>();

    // std::shared_ptr<pendulum>env;
    pol->to(torch::kDouble);
    cr->to(torch::kDouble);
    pol2->to(torch::kDouble);

    //---UNCOMENT THE ALGORITHM YOU WANT TO RUN---//
    algorithm* algo = new ac(pol, cr, env, 1e-4, 1e-3);
    // algorithm* algo = new dpg(pol2, cr, env, 2e-4, 1e-3);
    // algorithm* algo = new reinforce(pol, cr, env, 2e-4, 1e-3);

    for (int episode = 0; episode < 10000; episode++) {
        std::cout << "episode: " << episode << std::endl;
        algo->run_episode();
        algo->preprocess_states(algo->states);
        algo->step();

        std::cout << "NORM: " << torch::nn::utils::parameters_to_vector(pol->parameters()).norm().item<double>() << std::endl;
        std::cout << "sum of rewards: " << algo->sum_of_rewards << std::endl;
        // std::cout<<"returns: "<<algo->returns<<std::endl;
    }

    return 0;
}
