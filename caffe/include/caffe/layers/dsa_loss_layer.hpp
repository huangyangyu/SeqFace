#ifndef CAFFE_DSA_LOSS_LAYER_HPP_
#define CAFFE_DSA_LOSS_LAYER_HPP_

#include <vector>
#include <algorithm>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
/**
* @brief DSA loss layer.
*/
template <typename Dtype>
class DSALossLayer : public LossLayer<Dtype> {
 public:
  explicit DSALossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DSALoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return -1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  //    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  //    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // batch size
  int M_;
  // feature dimension
  int K_;
  // center number
  int N_;
  // identity face center number
  int faceid_num_;
  // sequence face center number
  int seqid_num_;
  // the actual number of center to use in each batch
  int ID_N_;

  // the parameters of dsaloss
  Dtype lambda_;
  Dtype prob_;
  Dtype gamma_;
  Dtype alpha_;
  Dtype beta_;

  // whether to ignore instances with a certain label
  bool has_ignore_label_;
  // the label indicating that an instance should be ignored
  int ignore_label_;
  // whether to normalize the center
  bool use_normalize_;
  // scale center agent when normalize center
  Dtype scale_;

  // dsaloss = centerloss + diffloss
  // use centerloss after center_iter_ times iteration
  int center_iter_;
  // use diffloss after diff_iter_ times iteration
  int diff_iter_;
  // the number of current iterations
  int iter_;

  Blob<Dtype> diff_;
  Blob<Dtype> square_diff_;
  Blob<Dtype> variation_sum_;
  Blob<int> id_vec_;
};

}  // namespace caffe

#endif  // CAFFE_DSA_LOSS_LAYER_HPP_
