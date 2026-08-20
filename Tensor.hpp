#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <unordered_set>
#include <functional>
#include <memory>
#include <random>

class Tensor {
public:
    std::vector<double> data;
    std::vector<double> grad;
    std::vector<size_t> shape;
    std::vector<size_t> strides;

    std::vector<Tensor*> parents;
    std::function<void()> backward_fn = [](){};

    Tensor(std::vector<double> val, std::vector<size_t> dimensions) {
        shape = dimensions;
        strides = compute_strides(shape);
        size_t expected = 1;

        for(size_t dim : dimensions) {
            expected *= dim;
        }
        
        if(expected != val.size()) {
            throw std::runtime_error("data size doesn't match shape");
        }

        data = val;
        grad = std::vector<double>(data.size(), 0.0); 
    }

    Tensor(std::vector<size_t> dimensions, std::vector<Tensor*> p = {}) {
        shape = dimensions;
        strides = compute_strides(shape);
        size_t total = 1;

        for(size_t s : shape) {
            total *= s;
        }

        data = std::vector<double>(total, 0.0);
        grad = std::vector<double>(total, 0.0);
        parents = p;
    }

    std::vector<size_t> compute_strides(const std::vector<size_t>& shape) {
        size_t rank = shape.size();
        std::vector<size_t> strides(rank);
        strides[rank - 1] = 1;

        for(int i = int(rank) - 2; i >= 0; i--)  {
                strides[i] = shape[i + 1] * strides[i + 1];
        }

        return strides;
    }

    std::vector<size_t> broadcast_shape(const std::vector<size_t>& a, const std::vector<size_t>& b) {
        size_t rank = std::max(a.size(), b.size());
        std::vector<size_t> out_shape(rank);

        for(size_t i = 0; i < rank; i++) {
            size_t dim_a = (i >= rank - a.size()) ? a[i - (rank - a.size())] : 1;
            size_t dim_b = (i >= rank - b.size()) ? b[i - (rank - b.size())] : 1;

            if(dim_a != dim_b && dim_a != 1 && dim_b != 1) {
                throw std::runtime_error("dimensions incompatible");
            }

            out_shape[i] = std::max(dim_a, dim_b);
        }

        return out_shape;
    }

    std::vector<size_t> broadcast_strides(const std::vector<size_t>& shape, const std::vector<size_t>& strides, const std::vector<size_t>& out_shape) {
        size_t out_rank = out_shape.size();
        size_t in_rank = shape.size();
        size_t padding = out_rank - in_rank;

        std::vector<size_t> result(out_rank);

        for(size_t i = 0; i < out_rank; i++) {
            if(i >= padding) {
                size_t real_axis = i - padding;
                if (shape[real_axis] == 1) {
                    result[i] = 0;
                } else {
                    result[i] = strides[real_axis];    
                }  
            }
            else {
                result[i] = 0;
            }
        }

        return result;
    }

    Tensor* add(Tensor &other) {
        std::vector<size_t> dimensions = broadcast_shape(this->shape, other.shape);
        Tensor* out = new Tensor(dimensions, {this, &other});
        std::vector<size_t> this_ostrides = broadcast_strides(this->shape, this->strides, out->shape);
        std::vector<size_t> other_ostrides = broadcast_strides(other.shape, other.strides, out->shape);
        
        size_t rank = out->shape.size();
        size_t total = out->data.size();
        std::vector<size_t> idx(rank, 0);

        for(size_t i = 0; i < total; i++) {
            size_t this_offset = 0;
            size_t other_offset = 0;
            size_t out_offset = 0;

            for(size_t k = 0; k < rank; k++) {
                this_offset += idx[k] * this_ostrides[k];
                other_offset += idx[k] * other_ostrides[k];
                out_offset += idx[k] * out->strides[k];
            }

            out->data[out_offset] = this->data[this_offset] + other.data[other_offset];

            for(int k = (int)rank - 1; k >= 0; k--) {
                idx[k]++;
                if (idx[k] >= out->shape[k]){
                    idx[k] = 0;
                }
                else {
                    break;
                }
            }
        }

        out->backward_fn = [this, &other, out, this_ostrides, other_ostrides, rank]() {
            size_t total = out->grad.size();
            std::vector<size_t> idx(rank, 0);

            for(size_t i = 0; i < total; i++) {
                size_t this_offset = 0;
                size_t other_offset = 0;
                size_t out_offset = 0;
                
                for(size_t k = 0; k < rank; k++) {
                    this_offset += idx[k] * this_ostrides[k];
                    other_offset += idx[k] * other_ostrides[k];
                    out_offset += idx[k] * out->strides[k];
                }

                this->grad[this_offset] += out->grad[out_offset];
                other.grad[other_offset] += out->grad[out_offset];

                for(int k = int(rank) - 1; k >= 0; k--) {
                    idx[k]++;
                    if(idx[k] >= out->shape[k]) {
                        idx[k] = 0;
                    }
                    else {
                        break;
                    }
                }
            }
        };

        return out;
    }
    
    Tensor* mul(Tensor &other) {
        std::vector<size_t> dimensions = broadcast_shape(this->shape, other.shape);
        Tensor* out = new Tensor(dimensions, {this, &other});
        std::vector<size_t> this_ostrides = broadcast_strides(this->shape, this->strides, out->shape);
        std::vector<size_t> other_ostrides = broadcast_strides(other.shape, other.strides, out->shape);
        
        size_t rank = out->shape.size();
        size_t total = out->data.size();
        std::vector<size_t> idx(rank, 0);

        for(size_t i = 0; i < total; i++) {
            size_t this_offset = 0;
            size_t other_offset = 0;
            size_t out_offset = 0;

            for(size_t k = 0; k < rank; k++) {
                this_offset += idx[k] * this_ostrides[k];
                other_offset += idx[k] * other_ostrides[k];
                out_offset += idx[k] * out->strides[k];
            }

            out->data[out_offset] = this->data[this_offset] * other.data[other_offset];

            for(int k = (int)rank - 1; k >= 0; k--) {
                idx[k]++;
                if (idx[k] >= out->shape[k]){
                    idx[k] = 0;
                }
                else {
                    break;
                }
            }
        }

        out->backward_fn = [this, &other, out, this_ostrides, other_ostrides, rank]() {
            size_t total = out->grad.size();
            std::vector<size_t> idx(rank, 0);

            for(size_t i = 0; i < total; i++) {
                size_t this_offset = 0;
                size_t other_offset = 0;
                size_t out_offset = 0;
                
                for(size_t k = 0; k < rank; k++) {
                    this_offset += idx[k] * this_ostrides[k];
                    other_offset += idx[k] * other_ostrides[k];
                    out_offset += idx[k] * out->strides[k];
                }

                this->grad[this_offset] += other.data[other_offset] * out->grad[out_offset];
                other.grad[other_offset] += this->data[this_offset] * out->grad[out_offset];

                for(int k = int(rank) - 1; k >= 0; k--) {
                    idx[k]++;
                    if(idx[k] >= out->shape[k]) {
                        idx[k] = 0;
                    }
                    else {
                        break;
                    }
                }
            }
        };

        return out;
    }

    Tensor* relu() {
        Tensor* out = new Tensor(this->shape, {this});
        size_t total = this->data.size();
        
        for(size_t i = 0; i < total; i++) {
            out->data[i] = (this->data[i] > 0) ? this->data[i] : 0;
        }

        out->backward_fn = [this, out, total]() {
            for(size_t i = 0; i < total; i++) {
                this->grad[i] += (this->data[i] > 0) ? out->grad[i] : 0;
            }
        };

        return out;
    }

    Tensor* matmul(Tensor &other) {
        std::vector<size_t> mat_dimensions(2);
        std::vector<size_t> this_batch = this->shape;
        std::vector<size_t> other_batch = other.shape;
        this_batch.resize(this->shape.size() - 2);
        other_batch.resize(other.shape.size() - 2);
        std::vector<size_t> batch_dimensions = broadcast_shape(this_batch, other_batch);

        size_t A_y = this->shape[this->shape.size() - 2];
        size_t A_x = this->shape[this->shape.size() - 1];
        size_t B_y = other.shape[other.shape.size() - 2];
        size_t B_x = other.shape[other.shape.size() - 1];

        if(A_x != B_y) {
            throw std::runtime_error("dimensions incompatible");
        }
        
        batch_dimensions.push_back(A_y);
        batch_dimensions.push_back(B_x);

        Tensor* out = new Tensor(batch_dimensions, {this, &other});
        std::vector<size_t> this_ostrides = broadcast_strides(this->shape, this->strides, out->shape);
        std::vector<size_t> other_ostrides = broadcast_strides(other.shape, other.strides, out->shape);
        
        size_t out_rank = out->shape.size();
        size_t this_rank = this->shape.size();
        size_t other_rank = other.shape.size();
        size_t batch_total = 1;

        for(int i = 0; i < out_rank - 2; i++) {
            batch_total *= out->shape[i];
        }

        std::vector<size_t> idx(out_rank, 0);

        for(size_t a = 0; a < batch_total; a++) {
            size_t this_offset = 0;
            size_t other_offset = 0;
            size_t out_offset = 0;

            for(size_t k = 0; k < out_rank - 2; k++) {
                this_offset += idx[k] * this_ostrides[k];
                other_offset += idx[k] * other_ostrides[k];
                out_offset += idx[k] * out->strides[k];
            }

            for(size_t i = 0; i < A_y; i++) {
                for(size_t j = 0; j < B_x; j++) {
                    for(size_t p = 0; p < A_x; p++) {
                        out->data[out_offset + (i * out->strides[out_rank - 2]) + (j * out->strides[out_rank-1])] += this->data[this_offset + (i * this->strides[this_rank - 2]) + (p * this->strides[this_rank - 1])] * other.data[other_offset + (p * other.strides[other_rank - 2]) + (j *  other.strides[other_rank - 1])];
                    }
                }
            }

            for(int k = (int)out_rank - 3; k >= 0; k--) {
                idx[k]++;
                if (idx[k] >= out->shape[k]){
                    idx[k] = 0;
                }
                else {
                    break;
                }
            }
        }

        out->backward_fn = [this, &other, out, this_ostrides, other_ostrides, batch_total, out_rank, other_rank, this_rank, A_y, A_x, B_y, B_x]() {
            std::vector<size_t> idx(out_rank, 0);

            for(size_t a = 0; a < batch_total; a++) {
                size_t this_offset = 0;
                size_t other_offset = 0;
                size_t out_offset = 0;

                for(size_t k = 0; k < out_rank - 2; k++) {
                    this_offset += idx[k] * this_ostrides[k];
                    other_offset += idx[k] * other_ostrides[k];
                    out_offset += idx[k] * out->strides[k];
                }

                for (size_t i = 0; i < A_y; i++) {
                    for (size_t p = 0; p < A_x; p++) {
                        double accumulate = 0.0;
                        for (size_t j = 0; j < B_x; j++) {
                            double dout_val = out->grad[out_offset + i*out->strides[out_rank-2] + j*out->strides[out_rank-1]];
                            double B_val = other.data[other_offset + p*other.strides[other_rank-2] + j*other.strides[other_rank-1]];
                            accumulate += dout_val * B_val;
                        }
                        this->grad[this_offset + i*this->strides[this_rank-2] + p*this->strides[this_rank-1]] += accumulate;
                    }
                }

                for (size_t p = 0; p < B_y; p++) {
                    for (size_t j = 0; j < B_x; j++) {
                        double accumulate = 0.0;
                        for (size_t i = 0; i < A_y; i++) {
                            double A_val = this->data[this_offset + i*this->strides[this_rank-2] + p*this->strides[this_rank-1]];
                            double dout_val = out->grad[out_offset + i*out->strides[out_rank-2] + j*out->strides[out_rank-1]];
                            accumulate += A_val * dout_val;
                        }
                        other.grad[other_offset + p*other.strides[other_rank-2] + j*other.strides[other_rank-1]] += accumulate;
                    }
                }
                for(int k = (int)out_rank - 3; k >= 0; k--) {
                    idx[k]++;
                    if (idx[k] >= out->shape[k]){
                        idx[k] = 0;
                    }
                    else {
                        break;
                    }
                }
            }
        };

        return out;
    }

    Tensor* sum() {
    Tensor* out = new Tensor(std::vector<size_t>{1}, {this});
    size_t total = this->data.size();
    out->data[0] = 0.0;

    for (size_t i = 0; i < total; i++) {
        out->data[0] += this->data[i];
    }

    out->backward_fn = [this, out, total]() {
        for (size_t i = 0; i < total; i++) {
            this->grad[i] += out->grad[0];
        }
    };

    return out;
}

    void build_topo(Tensor* v, std::unordered_set<Tensor*>& visited, std::vector<Tensor*>& topo) {
        if(visited.find(v) == visited.end()) {
            visited.insert(v);
            for(Tensor* parent: v->parents) {
                build_topo(parent, visited, topo);
            }
            topo.push_back(v);
        }
    }

    void backward() {
        std::vector<Tensor*> topo;
        std::unordered_set<Tensor*> visited;
        build_topo(this, visited, topo);
        
        grad[0] = 1.0;

        for(auto i = topo.rbegin(); i != topo.rend(); ++i) {
            (*i)->backward_fn();
        }
    }

    void zero_grad() {
        for(auto &i : grad) {
            i = 0.0;
        }
    }

};