#ifndef CAFFE_TRIPLET_LOSS_LAYER_HPP_
#define CAFFE_TRIPLET_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/** 
 * @brief Computes the triplet loss 
 */  
template <typename Dtype>  
class TripletLossLayer : public LossLayer<Dtype> {  
 public:  
  explicit TripletLossLayer(const LayerParameter& param)  
      : LossLayer<Dtype>(param){}  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top);  
  
  virtual inline int ExactNumBottomBlobs() const { return 4; }  
  //virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline const char* type() const { return "TripletLoss"; }  
  /** 
   * Unlike most loss layers, in the TripletLossLayer we can backpropagate 
   * to the first three inputs. 
   */  
  virtual inline bool AllowForceBackward(const int bottom_index) const {  
    return bottom_index != 3;  
  }  
  
 protected:  
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top);  
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,  
      const vector<Blob<Dtype>*>& top);  
  
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,  
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);  
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,  
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);  
  
  Blob<Dtype> diff_ap_;  // cached for backward pass  
  Blob<Dtype> diff_an_;  // cached for backward pass  
  Blob<Dtype> diff_pn_;  // cached for backward pass  
  
  Blob<Dtype> diff_sq_ap_;  // cached for backward pass  
  Blob<Dtype> diff_sq_an_;  // tmp storage for gpu forward pass  
  
  Blob<Dtype> dist_sq_ap_;  // cached for backward pass  
  Blob<Dtype> dist_sq_an_;  // cached for backward pass  
  
  Blob<Dtype> summer_vec_;  // tmp storage for gpu forward pass  
  Blob<Dtype> dist_binary_;  // tmp storage for gpu forward pass  
};

}  // namespace caffe

#endif  // CAFFE_TRIPLET_LOSS_LAYER_HPP_
