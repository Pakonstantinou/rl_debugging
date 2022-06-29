#include "actor_critic.h"
#include "dpg.h"
#include "reinforce.h"

struct MyPolicy : public stochastic {
public:
    torch::nn::Linear fc1, out, out_logsigma;

    MyPolicy() : fc1(register_module("fc1", torch::nn::Linear(2, 256))),
                 out(register_module("out", torch::nn::Linear(256, 1))),
                 out_logsigma(register_module("out_sigma", torch::nn::Linear(256, 1))) {}

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x)
    {
        x = torch::relu(fc1(x));
        torch::Tensor mu = out(x);
        torch::Tensor logsigma = out_logsigma(x);

/*        auto muIsNan = at::isnan(mu).any().item<bool>();

        if (muIsNan==1){
            std::cout<<"mu"<<mu<<std::endl;
            std::cout<<"mu nan"<<std::endl;

            std::cout << "ac parameters: " << this->parameters() << std::endl;

               std::cout << "weights fci: " << fc1->weight << std::endl;
                std::cout << "weights out: " << out->weight << std::endl;
                std::cout << "weights outlogsigma: " <<out_logsigma->weight << std::endl;

            std::cout << "weights fci .grad(): " << fc1->weight.grad() << std::endl;
            std::cout << "weights out .grad(): " << out->weight.grad() << std::endl;
            std::cout << "weights outlogsigma: .grad() " <<out_logsigma->weight.grad() << std::endl;
            exit (EXIT_FAILURE);
        }
        auto logsigmaIsNan = at::isnan(logsigma).any().item<bool>();

        if (logsigmaIsNan==1){
            std::cout<<"logsigma"<<logsigma<<std::endl;
            std::cout<<"logsigma nan"<<std::endl;

            std::cout << "ac parameters: " << this->parameters() << std::endl;

            std::cout << "weights fci: " << fc1->weight << std::endl;
            std::cout << "weights out: " << out->weight << std::endl;
            std::cout << "weights outlogsigma: " <<out_logsigma->weight << std::endl;

            std::cout << "weights fci .grad(): " << fc1->weight.grad() << std::endl;
            std::cout << "weights out .grad(): " << out->weight.grad() << std::endl;
            std::cout << "weights outlogsigma: .grad() " <<out_logsigma->weight.grad() << std::endl;
            exit (EXIT_FAILURE);
        }*/
        return {mu, logsigma};
    }
};

struct MyPolicy2 : public deterministic {
public:
    torch::nn::Linear fc1, out, out_logsigma;

    MyPolicy2() : fc1(register_module("fc1", torch::nn::Linear(2, 128))),
                  out(register_module("out", torch::nn::Linear(128, 1))),
                  out_logsigma(register_module("out_sigma", torch::nn::Linear(128, 1))) {}

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x)
    {
        x = torch::relu(fc1(x));
        torch::Tensor mu = out(x);
        torch::Tensor logsigma = out_logsigma(x);

        return {mu, logsigma};
    }
};

struct MyCritic : public criticgeneral {
public:
    torch::nn::Linear fc1, out;

    MyCritic() : fc1(register_module("fc1", torch::nn::Linear(3, 256))),
                 out(register_module("out", torch::nn::Linear(256, 1))) {}

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
    algorithm* algo = new ac (pol, cr, env, 0.0005, 0.001);
     //algorithm* algo = new dpg(pol2, cr, env, 2e-4, 1e-3);
    // algorithm* algo = new reinforce(pol, cr, env, 2e-4, 1e-3);

    for (int episode = 0; episode < 1000000; episode++) {
        std::cout << "episode: " << episode << std::endl;
        algo->run_episode();
        algo->preprocess_states(algo->states);
        algo->step();
        //std::cout <<algo->states<<std::endl;
        //std::cout <<torch::cat({algo->actions.unsqueeze(1), algo->rewards.unsqueeze(1)}, 1)<< std::endl;
        //std::cout << "NORM: " << torch::nn::utils::parameters_to_vector(pol->parameters()).norm().item<double>() << std::endl;
        std::cout << "sum of rewards: " << algo->sum_of_rewards << std::endl;
        // std::cout<<"returns: "<<algo->returns<<std::endl;
    }

    return 0;
}
