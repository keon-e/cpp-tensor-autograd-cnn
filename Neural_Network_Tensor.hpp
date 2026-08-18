#include "Tensor.hpp"

class Layer {
    public:
        Tensor* weights;
        Tensor* bias;

    Layer(int nin, int nout) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        std::vector<double> w_data((size_t)nin * (size_t)nout);

        for (auto& w : w_data) {
            w = dist(gen);
        }
        
        weights = new Tensor(w_data, std::vector<size_t>{(size_t)nin, (size_t)nout});

        std::vector<double> b_data((size_t)nout, 0.0);
        bias = new Tensor(b_data, std::vector<size_t>{(size_t)nout});
    }

    Tensor* forward(Tensor* input) {
        Tensor* out = input->matmul(*weights);
        Tensor* out_biased = out->add(*bias);
        return out_biased->relu();
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

    Tensor* forward(Tensor* input) {
        Tensor* current = input;

        for(size_t i = 0; i < layers.size(); i++) {
            current = layers[i].forward(current);
        }

        return current;
    }

    void zero_grad_mlp() {
        for(auto& layer : layers) {
            layer.zero_grad_layer();
        }
    }

};