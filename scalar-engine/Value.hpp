#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <unordered_set>
#include <functional>
#include <memory>
#include <random>

class Value {
public:

    double data;
    double grad;
    std::vector<Value*> parents;
    std::function<void()> backward_fn = [](){};

    Value(double val) {
        data = val;
        grad = 0.0;
    }

    Value(double val, std::vector<Value*> p) {
        data = val;
        grad = 0.0;
        parents = p;
    }

    Value* add(Value& other) {
        double result = data + other.data;
        Value* out = new Value(result, {this, &other});

        out->backward_fn = [this, &other, out]() {
            grad += out->grad;
            other.grad += out->grad;
        };

        return out;
    }

    Value* multiply(Value& other) {
        double result = data * other.data;
        Value* out = new Value(result, {this, & other});

        out->backward_fn = [this, &other, out]() {
            grad += other.data * out->grad;
            other.grad += data * out->grad;
        };

        return out;
    }

    Value* pow(double exponent) {
        double result = std::pow(data, exponent);
        Value *out = new Value(result, {this});

        out->backward_fn = [this, out, exponent]() {
            grad += (exponent * std::pow(data, exponent - 1)) * out->grad;
        };

        return out;
    }

    Value* negate() {
        Value* neg_self = new Value(-1.0);
        return multiply(*neg_self);
    }

    Value* subtract(Value& other) {
        Value* neg_other = other.negate();
        return add(*neg_other);
    }

    Value* divide(Value& other) {
        Value* reciprocal_other = other.pow(-1);
        return multiply(*reciprocal_other);
    }

    Value* relu() {
        double result = (data < 0.0) ? 0.0 : data;
        Value* out = new Value(result, {this});

        out->backward_fn = [this, out]() {
            grad += (data < 0.0) ? 0.0 : out->grad;
        };

        return out;
    }

    void build_topo(Value *v, std::unordered_set<Value*> &visited, std::vector<Value*> &topo) {

        if(visited.find(v) == visited.end()) {

            visited.insert(v);

            for(Value* parent : v->parents) {
                build_topo(parent, visited, topo);
            }

            topo.push_back(v);
        }

    }

    void backward() {
        std::vector<Value*> topo;
        std::unordered_set<Value*> visited;

        build_topo(this, visited, topo);

        grad = 1.0;

        for (auto cur = topo.rbegin(); cur != topo.rend(); ++cur) {
            Value* v = *cur;
            v->backward_fn();
        }
    }

     
};

Value* operator+(Value& a, Value& b) { 
    return a.add(b); 
}

Value* operator*(Value& a, Value& b) { 
    return a.multiply(b); 
}

Value* operator-(Value& a, Value& b) { 
    return a.subtract(b); 
}

Value* operator/(Value& a, Value& b) { 
    return a.divide(b); 
}

Value* operator^(Value& a, double exponent) { 
    return a.pow(exponent); 
}
