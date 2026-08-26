#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "Tensor.hpp"

class MNIST_Data {
public:
    std::vector<std::vector<double>> images;
    std::vector<int> numbers;

    MNIST_Data(const std::string& path) {
        std::ifstream file(path);

        if(!file.is_open()) {
            throw std::runtime_error("could not open file");
        }

        std::string current;
        std::getline(file, current);

        while(std::getline(file, current)) {
            std::stringstream line(current);
            std::string value;

            std::getline(line, value, ',');
            int number = std::stoi(value);

            std::vector<double> pixels;
            pixels.reserve(784);

            while(getline(line, value, ',')) {
                pixels.push_back(std::stod(value) / 255.0);
            }

            numbers.push_back(number);
            images.push_back(pixels);

        }
    }

    std::shared_ptr<Tensor> batch_image(size_t start, size_t batch_size) {
        std::vector<double> flat_data;
        flat_data.reserve(batch_size * 784);

        for(size_t i = start; i < start + batch_size; i++) {
            for(auto k : images[i]) {
                flat_data.push_back(k);
            }
        }

        return std::make_shared<Tensor>(flat_data, std::vector<size_t>{batch_size, 784});
    }

    std::shared_ptr<Tensor> batch_image_2d(size_t start, size_t batch_size) {
        std::vector<double> flat_data;
        flat_data.reserve(batch_size * 784);

        for(size_t i = start; i < start + batch_size; i++)
            for(auto k : images[i]) 
                flat_data.push_back(k);
                
        return std::make_shared<Tensor>(flat_data, std::vector<size_t>{batch_size, 1, 28, 28});
    }

    std::shared_ptr<Tensor> batch_target(size_t start, size_t batch_size) {
        std::vector<double> one_hot(batch_size * 10, 0.0);

        for(size_t i = start; i < batch_size + start; i++) {
            int value = numbers[i];
            one_hot[(i - start) * 10 + value] = 1.0;
        }

        return std::make_shared<Tensor>(one_hot, std::vector<size_t>{batch_size, 10});
    }

};