#include <cmath>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/recurrent_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class RecurrentLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  RecurrentLayerTest()
      : blob_bottom_1_(new Blob<Dtype>(10, 5, 2, 3)),
        blob_bottom_2_(new Blob<Dtype>(10, 5, 2, 3)),
        blob_bottom_3_(new Blob<Dtype>(10, 5, 2, 3)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_1_);
    filler.Fill(this->blob_bottom_2_);
    filler.Fill(this->blob_bottom_3_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_bottom_vec_.push_back(blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_3_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~RecurrentLayerTest() {
    delete blob_bottom_1_;
    delete blob_bottom_2_;
    delete blob_bottom_3_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_bottom_3_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(RecurrentLayerTest, TestDtypesAndDevices);

TYPED_TEST(RecurrentLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  //RecurrentLayer<Dtype> layer(layer_param);
  //GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  //checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
  //    this->blob_top_vec_);
}

}  // namespace caffe
