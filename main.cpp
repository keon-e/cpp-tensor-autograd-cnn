#include <iostream>
#include <vector>
#include <random>
#include "Tensor.hpp"
#include "Neural_Network_Tensor.hpp"

Tensor* mse_loss(Tensor* predictions, Tensor& targets) {
    std::vector<double> negative_data(targets.data.size());
    
    for(int i = 0; i < targets.data.size(); i++) {
        negative_data[i] = -targets.data[i];
    }

    Tensor negative_targets = Tensor(negative_data, targets.shape);
    Tensor* diff = predictions->add(negative_targets);
    Tensor* diff_squared = diff->mul(*diff);

    return diff_squared->sum();
}

int main() {
    // --- Tiny synthetic problem: learn to map a 4-dim input to a 2-dim target ---
    MLP mlp(std::vector<size_t>{4, 5, 2});

    Tensor input(std::vector<double>{1.0, 0.5, -1.0, 2.0}, std::vector<size_t>{1, 4});
    Tensor target(std::vector<double>{1.0, 0.0}, std::vector<size_t>{1, 2});

    double learning_rate = 0.01;
    int num_steps = 200;

    for (int step = 0; step < num_steps; step++) {
        // 1. zero gradients from the previous step
        mlp.zero_grad_mlp();

        // 2. forward pass
        Tensor* predictions = mlp.forward(&input);

        // 3. compute loss
        Tensor* loss = mse_loss(predictions, target);

        // 4. backward pass
        loss->backward();

        // 5. update weights (plain gradient descent, outside the graph)
        for (auto& layer : mlp.layers) {
            for (size_t i = 0; i < layer.weights->data.size(); i++)
                layer.weights->data[i] -= learning_rate * layer.weights->grad[i];
            for (size_t i = 0; i < layer.bias->data.size(); i++)
                layer.bias->data[i] -= learning_rate * layer.bias->grad[i];
        }

        if (step % 20 == 0) {
            std::cout << "step " << step << " loss: " << loss->data[0] << std::endl;
        }
    }

    // final predictions, should be closer to target = [1.0, 0.0] than at the start
    Tensor* final_pred = mlp.forward(&input);
    std::cout << "final predictions: ";
    for (double v : final_pred->data) std::cout << v << " ";
    std::cout << std::endl;
    std::cout << "target: 1 0" << std::endl;

    return 0;
}