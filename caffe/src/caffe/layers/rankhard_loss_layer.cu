#include <algorithm>
#include <vector>
#include <cfloat>
#include <cmath>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/rankhard_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void RankHardLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Forward_cpu(bottom, top);
}

template <typename Dtype>
void RankHardLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
   Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_LAYER_GPU_FUNCS(RankHardLossLayer);

}  // namespace caffe
