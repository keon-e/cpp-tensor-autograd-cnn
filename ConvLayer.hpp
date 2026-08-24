#include "Tensor.hpp"
#include "Neural_Network_Tensor.hpp"

class ConvLayer {
public:
    Tensor* filter;
    Tensor* bias;

    ConvLayer();

    ConvLayer(size_t filter_h, size_t filter_w) {
        std::random_device rd;
        std::mt19937 gen(rd());
        double limit = 1.0 / std::sqrt((double)(filter_h * filter_w));
        std::uniform_real_distribution<double> dist(-limit, limit);

        std::vector<double> f_data(filter_h * filter_w);
        for(auto& f : f_data) {
            f = dist(gen);
        }
        filter = new Tensor(f_data, std::vector<size_t>{filter_h, filter_w});

        std::vector<double> b_data(1, 0.0);
        bias = new Tensor(b_data, std::vector<size_t>{1});
    }

    Tensor* forward(Tensor* input) {
        Tensor* out = input->conv2d(*filter);
        Tensor* out_biased = out->add(*bias);
        return out_biased->relu();
    }

    void zero_grad_conv() {
        filter->zero_grad();
        bias->zero_grad();
    }
};

class CNN {
public:
    ConvLayer conv1;
    ConvLayer conv2;
    MLP mlp;

    CNN(size_t filter_h, size_t filter_w, std::vector<size_t> mlp_nouts) {
        conv1 = ConvLayer(filter_h, filter_w);
        conv2 = ConvLayer(filter_h, filter_w);
        mlp = MLP(mlp_nouts);
    }

    Tensor* forward(Tensor* input) {
        Tensor* conv_out_1 = conv1.forward(input);
        Tensor* pooled_1 = conv_out_1->maxpool2d(2);

        Tensor* conv_out_2 = conv2.forward(pooled_1);
        Tensor* pooled_2 = conv_out_2->maxpool2d(2);

        Tensor* flat = pooled_2->reshape(std::vector<size_t>{pooled_2->shape[0], pooled_2->shape[1] * pooled_2->shape[2]});

        return mlp.forward(flat);
    }

};