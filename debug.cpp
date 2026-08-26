#include <iostream>
#include "Tensor.hpp"
#include "Neural_Network_Tensor.hpp"
#include "Conv_Layer.hpp"

int main() {
    CNN cnn(3, 3, std::vector<size_t>{4, 8}, std::vector<size_t>{200, 32, 10});

    std::shared_ptr<Tensor> x = std::make_shared<Tensor>(std::vector<double>(1*1*28*28, 0.5), std::vector<size_t>{1, 1, 28, 28});
    std::shared_ptr<Tensor> target = std::make_shared<Tensor>(std::vector<double>{1,0,0,0,0,0,0,0,0,0}, std::vector<size_t>{1, 10});

    for (int step = 0; step < 20; step++) {
        cnn.zero_grad_cnn();

        std::shared_ptr<Tensor> pred = cnn.forward(x);

        std::vector<double> neg(10);
        for (int i = 0; i < 10; i++) neg[i] = -target->data[i];
        std::shared_ptr<Tensor> neg_t = std::make_shared<Tensor>(neg, std::vector<size_t>{1,10});
        std::shared_ptr<Tensor> diff = pred->add(neg_t);
        std::shared_ptr<Tensor> loss = diff->mul(diff)->sum();

        loss->backward();

        double conv1_grad_sum = 0, conv2_grad_sum = 0, mlp0_grad_sum = 0;
        for (auto v : cnn.conv1.filter->grad) conv1_grad_sum += std::abs(v);
        for (auto v : cnn.conv2.filter->grad) conv2_grad_sum += std::abs(v);
        for (auto v : cnn.mlp.layers[0].weights->grad) mlp0_grad_sum += std::abs(v);

        std::cout << "step " << step << " loss: " << loss->data[0]
                   << " | conv1: " << conv1_grad_sum
                   << " conv2: " << conv2_grad_sum
                   << " mlp0: " << mlp0_grad_sum << std::endl;

        for (size_t i = 0; i < cnn.conv1.filter->data.size(); i++)
            cnn.conv1.filter->data[i] -= 0.01 * cnn.conv1.filter->grad[i];
        for (size_t i = 0; i < cnn.conv2.filter->data.size(); i++)
            cnn.conv2.filter->data[i] -= 0.01 * cnn.conv2.filter->grad[i];
        for (auto& layer : cnn.mlp.layers)
            for (size_t i = 0; i < layer.weights->data.size(); i++)
                layer.weights->data[i] -= 0.01 * layer.weights->grad[i];
    }

    return 0;
}