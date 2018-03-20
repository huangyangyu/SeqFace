#ifndef CAFFE_T_DSA_LOSS_LAYER_HPP_
#define CAFFE_T_DSA_LOSS_LAYER_HPP_

#include <vector>
#include <algorithm>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
/**
* @brief T_DSA loss layer.
*/
template <typename Dtype>
class T_DSALossLayer : public LossLayer<Dtype> {
 public:
  explicit T_DSALossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int M_;
  int K_;
  int N_;
  int faceid_num_;
  int seqid_num_;
  int ID_N_;
  Dtype lambda_;
  Dtype prob_;
  unsigned int uint_prob_;
  Dtype gamma_;
  Dtype alpha_;
  Dtype beta_;
  /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
  bool use_normalize_;
  Dtype scale_;
  int center_iter_;
  int diff_iter_;
  int iter_;

  Blob<Dtype> diff_;
  Blob<Dtype> square_diff_;
  Blob<Dtype> variation_sum_;
  Blob<Dtype> valid_counts_;
  Blob<int> count_;
  Blob<unsigned int> rand_vec_;
  Blob<int> id_vec_;
  Blob<Dtype> loss_;
  Blob<Dtype> loss1_;
  Blob<Dtype> loss2_;
};

}  // namespace caffe

#endif  // CAFFE_T_DSA_LOSS_LAYER_HPP_
