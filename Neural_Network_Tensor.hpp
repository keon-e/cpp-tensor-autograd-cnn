#pragma once

#include "Tensor.hpp"

class Layer {
    public:
        std::shared_ptr<Tensor> weights;
        std::shared_ptr<Tensor> bias;

    Layer(int nin, int nout) {
        std::random_device rd;
        std::mt19937 gen(rd());
        double bound = 1.0 / std::sqrt((double)nin);
        std::uniform_real_distribution<double> dist(-bound, bound);
        std::vector<double> w_data((size_t)nin * (size_t)nout);

        for (auto& w : w_data) {
            w = dist(gen);
        }
        
        weights = std::make_shared<Tensor>(w_data, std::vector<size_t>{(size_t)nin, (size_t)nout});

        std::vector<double> b_data((size_t)nout, 0.0);
        bias = std::make_shared<Tensor>(b_data, std::vector<size_t>{(size_t)nout});
    }

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) {
        std::shared_ptr<Tensor> out = input->matmul(weights);
        std::shared_ptr<Tensor> out_biased = out->add(bias);
        return out_biased->relu();
    }

    std::shared_ptr<Tensor> forward_no_relu(std::shared_ptr<Tensor> input) {
        std::shared_ptr<Tensor> out = input->matmul(weights);
        return out->add(bias);
    }

    void zero_grad_layer() {
        weights->zero_grad();
        bias->zero_grad();
    }
};

class MLP {
public:
    std::vector<Layer> layers;

    MLP(std::vector<size_t> nouts) {

        for(size_t i = 0; i < nouts.size() - 1; i++) {
                layers.push_back(Layer(nouts[i], nouts[i + 1]));
        }
    }

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) {
        std::shared_ptr<Tensor> current = input;

        for(size_t i = 0; i < layers.size() - 1; i++) {
            current = layers[i].forward(current);
        }
        
        current = layers[layers.size() - 1].forward_no_relu(current);
        return current;
    }

    void zero_grad_mlp() {
        for(auto& layer : layers) {
            layer.zero_grad_layer();
        }
    }

};