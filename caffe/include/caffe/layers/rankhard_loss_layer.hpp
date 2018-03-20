#ifndef CAFFE_RANK_HARD_LOSS_LAYER_HPP_
#define CAFFE_RANK_HARD_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class RankHardLossLayer : public LossLayer<Dtype> {
 public:
  explicit RankHardLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RankHardLoss"; }

  virtual inline bool AllowForceBackward(const int bottom_index) const { return true; }

  protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  void set_mask(const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> mask_;
  Blob<Dtype> dis_;
  Blob<Dtype> diff_;
};
}  // namespace caffe

#endif  // CAFFE_RANK_HARD_LOSS_LAYER_HPP_
