#pragma once

#include "Tensor.hpp"
#include "Neural_Network_Tensor.hpp"

class ConvLayer {
public:
    std::shared_ptr<Tensor> filter;
    std::shared_ptr<Tensor> bias;

    ConvLayer(size_t num_filters, size_t channels, size_t filter_h, size_t filter_w) {
        std::random_device rd;
        std::mt19937 gen(rd());
        double limit = 1.0 / std::sqrt((double)(filter_h * filter_w));
        std::uniform_real_distribution<double> dist(-limit, limit);

        std::vector<double> f_data(num_filters * channels * filter_h * filter_w);
        for(auto& f : f_data) {
            f = dist(gen);
        }
        filter = std::make_shared<Tensor>(f_data, std::vector<size_t>{num_filters, channels, filter_h, filter_w});

        std::vector<double> b_data(num_filters, 0.0);
        bias = std::make_shared<Tensor>(b_data, std::vector<size_t>{num_filters, 1, 1});
    }

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) {
        std::shared_ptr<Tensor> out = input->conv2d(filter);
        std::shared_ptr<Tensor> out_biased = out->add(bias);
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

    CNN(size_t filter_h, size_t filter_w, std::vector<size_t> num_filters, std::vector<size_t> mlp_nouts) : conv1(num_filters[0], 1, filter_h, filter_w), conv2(num_filters[1], num_filters[0], filter_h, filter_w), mlp(mlp_nouts) {}

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> input) {
        std::shared_ptr<Tensor> conv_out_1 = conv1.forward(input);
        std::shared_ptr<Tensor> pooled_1 = conv_out_1->maxpool2d(2);

        std::shared_ptr<Tensor> conv_out_2 = conv2.forward(pooled_1);
        std::shared_ptr<Tensor> pooled_2 = conv_out_2->maxpool2d(2);

        std::shared_ptr<Tensor> flat = pooled_2->reshape(std::vector<size_t>{pooled_2->shape[0], pooled_2->shape[1] * pooled_2->shape[2] * pooled_2->shape[3]});

        return mlp.forward(flat);
    }

    void zero_grad_cnn() {
        conv1.zero_grad_conv();
        conv2.zero_grad_conv();
        mlp.zero_grad_mlp();
    }
};