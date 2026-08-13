#include <Value.hpp>

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
                if (idx[k] < out->shape[k]){
                    break;
                }
                idx[k] = 0;
            }
        }
        return out;
    }
    
};

