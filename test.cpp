#include <iostream>
#include "Tensor.hpp"
#include "Neural_Network_Tensor.hpp"
#include "Conv_Layer.hpp"
#include "CSV_Parser.hpp"

std::shared_ptr<Tensor> mse_loss(std::shared_ptr<Tensor> predictions, std::shared_ptr<Tensor> targets) {
    std::vector<double> negative_data(targets->data.size());
    for (size_t i = 0; i < targets->data.size(); i++) {
        negative_data[i] = -targets->data[i];
    }
    std::shared_ptr<Tensor> negative_targets = std::make_shared<Tensor>(negative_data, targets->shape);
    std::shared_ptr<Tensor> diff = predictions->add(negative_targets);
    std::shared_ptr<Tensor> diff_squared = diff->mul(diff);
    return diff_squared->sum();
}

double check_accuracy(CNN& cnn, MNIST_Data& val_data, size_t batch_size) {
    size_t num_batches = val_data.images.size() / batch_size;
    size_t correct = 0;
    size_t total = 0;

    for (size_t batch = 0; batch < num_batches; batch++) {
        size_t start = batch * batch_size;
        std::shared_ptr<Tensor> input = val_data.batch_image_2d(start, batch_size);
        std::shared_ptr<Tensor> predictions = cnn.forward(input);

        for (size_t i = 0; i < batch_size; i++) {
            size_t best_idx = 0;
            double best_val = predictions->data[i * 10];
            for (size_t c = 1; c < 10; c++) {
                if (predictions->data[i * 10 + c] > best_val) {
                    best_val = predictions->data[i * 10 + c];
                    best_idx = c;
                }
            }
            if ((int)best_idx == val_data.numbers[start + i]) correct++;
            total++;
        }
    }

    return (double)correct / total;
}

int main() {
    std::cout << "loading data..." << std::endl;
    MNIST_Data train_data("mnist_train.csv");
    MNIST_Data val_data("mnist_val.csv");
    std::cout << "loaded " << train_data.images.size() << " training images" << std::endl;

    CNN cnn(3, 3, std::vector<size_t>{4, 8}, std::vector<size_t>{25*8, 32, 10});

    double learning_rate = 0.01;
    size_t batch_size = 32;
    int num_epochs = 10;

    size_t num_batches = train_data.images.size() / batch_size;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        for (size_t batch = 0; batch < num_batches; batch++) {
            size_t start = batch * batch_size;

            std::shared_ptr<Tensor> input = train_data.batch_image_2d(start, batch_size);
            std::shared_ptr<Tensor> target = train_data.batch_target(start, batch_size);

            cnn.zero_grad_cnn();

            std::shared_ptr<Tensor> predictions = cnn.forward(input);
            std::shared_ptr<Tensor> loss = mse_loss(predictions, target);
            loss->backward();

            for (auto* layer_conv : std::vector<ConvLayer*>{&cnn.conv1, &cnn.conv2}) {
                for (size_t i = 0; i < layer_conv->filter->data.size(); i++)
                    layer_conv->filter->data[i] -= learning_rate * layer_conv->filter->grad[i];
                for (size_t i = 0; i < layer_conv->bias->data.size(); i++)
                    layer_conv->bias->data[i] -= learning_rate * layer_conv->bias->grad[i];
            }

            for (auto& layer : cnn.mlp.layers) {
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

    double acc = check_accuracy(cnn, val_data, batch_size);
    std::cout << "validation accuracy: " << (acc * 100.0) << "%" << std::endl;

    return 0;
}