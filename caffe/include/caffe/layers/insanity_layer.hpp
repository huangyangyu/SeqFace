#ifndef CAFFE_INSANITY_LAYER_HPP_
#define CAFFE_INSANITY_LAYER_HPP_

//#include <string>
//#include <utility>
#include <vector>

//#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/layer.hpp"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {
/**
* @brief Randomized Leaky Rectified Linear Unit @f$
*        y_i = \max(0, x_i) + frac{\min(0, x_i)}{a_i}
*        @f$. The negative slope is randomly generated from
*        uniform distribution U(lb, ub).
*/
template <typename Dtype>
class InsanityLayer : public NeuronLayer<Dtype> {
public:
/**
* @param param provides InsanityParameter insanity_param,
*/
explicit InsanityLayer(const LayerParameter& param)
    : NeuronLayer<Dtype>(param) {}

virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

virtual inline const char* type() const { return "Insanity"; }

protected:
/**
* @param bottom input Blob vector (length 1)
*   -# @f$ (N \times C \times ...) @f$
*      the inputs @f$ x @f$
* @param top output Blob vector (length 1)
*   -# @f$ (N \times C \times ...) @f$
*      the computed outputs for each channel @f$i@f$ @f$
*        y_i = \max(0, x_i) + a_i \min(0, x_i)
*      @f$.
*/
virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


Dtype lb_, ub_, mean_slope;
Blob<Dtype> alpha;  // random generated negative slope
Blob<Dtype> bottom_memory_;  // memory for in-place computation
};

}  // namespace caffe

#endif  // CAFFE_INSANITY_LAYER_HPP_
