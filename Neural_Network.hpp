#include <Value.hpp>

class Neuron {
    public:
        std::vector<Value*> weights;
        Value* bias;

    Neuron(int nin) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for(int i = 0; i < nin; i++)
            weights.push_back(new Value(dist(gen)));

        bias = new Value(0);
    }

    Value *forward(std::vector<Value*> inputs) {

        Value *total = bias;

        for(int i = 0; i < inputs.size(); i++) {
            Value* product = *inputs[i] * *weights[i];
            total = *total + *product;
        }

        return total->relu();
    }

    
};

class Layer {
    public:
        std::vector<Neuron> neurons;

    Layer(int nin, int nout) {

        for(int i = 0; i < nout; i++)
            neurons.push_back(Neuron(nin));
    }

    std::vector<Value*> forward(std::vector<Value*> inputs) {
        std::vector<Value*> outputs;
        
        for(int i = 0; i < neurons.size(); i++)
            outputs.push_back(neurons[i].forward(inputs));
        
        return outputs;
    }
};

class MLP {
    public:
        std::vector<Layer> layers;

        MLP(int nin, std::vector<int> nouts) {
            layers.push_back(Layer(nin, nouts[0]));

            for(int i = 1; i < nouts.size(); i++)
                layers.push_back(Layer(nouts[i-1], nouts[i]));
        }

        std::vector<Value*> forward(std::vector<Value*> inputs) {
            std::vector<Value*> current = inputs;

            for(int i = 0; i < layers.size(); i++) {
                current = layers[i].forward(current);
            }
            return current;
        }
};