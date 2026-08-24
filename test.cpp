#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "Tensor.hpp"
#include "Neural_Network_Tensor.hpp"
#include "CSV_Parser.hpp"

Tensor* mse_loss(Tensor* predictions, Tensor& targets) {
    std::vector<double> negative_data(targets.data.size());

    for (size_t i = 0; i < targets.data.size(); i++) {
        negative_data[i] = -targets.data[i];
    }

    Tensor* negative_targets = new Tensor(negative_data, targets.shape);
    Tensor* diff = predictions->add(*negative_targets);
    Tensor* diff_squared = diff->mul(*diff);

    return diff_squared->sum();
}

int main() {
    std::cout << "loading data..." << std::endl;
    MNIST_Data train_data("mnist_train.csv");
    std::cout << "loaded " << train_data.images.size() << " training images" << std::endl;

    MLP mlp(std::vector<size_t>{784, 128, 64, 10});

    double learning_rate = 0.01;
    size_t batch_size = 32;
    int num_epochs = 10;

    size_t num_batches = train_data.images.size() / batch_size;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        for (size_t batch = 0; batch < num_batches; batch++) {
            size_t start = batch * batch_size;

            Tensor* input = train_data.batch_image(start, batch_size);
            Tensor* target = train_data.batch_target(start, batch_size);

            mlp.zero_grad_mlp();

            Tensor* predictions = mlp.forward(input);
            Tensor* loss = mse_loss(predictions, *target);
            loss->backward();

            for (auto& layer : mlp.layers) {
                for (size_t i = 0; i < layer.weights->data.size(); i++)
                    layer.weights->data[i] -= learning_rate * layer.weights->grad[i];
                for (size_t i = 0; i < layer.bias->data.size(); i++)
                    layer.bias->data[i] -= learning_rate * layer.bias->grad[i];
            }

            if (batch % 200 == 0) {
                std::cout << "epoch " << epoch << " batch " << batch << " loss: " << loss->data[0] << std::endl;
            }
        }
    }

    std::cout << "training done" << std::endl;

    return 0;
}