#include <algorithm>
#include <vector>

#include "caffe/layers/lifted_struct_similarity_softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LiftedStructSimilaritySoftmaxLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void LiftedStructSimilaritySoftmaxLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

INSTANTIATE_LAYER_GPU_FUNCS(LiftedStructSimilaritySoftmaxLossLayer);
}  // namespace caffe

