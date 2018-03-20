#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/normalize_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class NormalizeLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  NormalizeLayerTest()
      : blob_bottom_(new Blob<Dtype>(10, 4, 2, 3)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~NormalizeLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(NormalizeLayerTest, TestDtypesAndDevices);

TYPED_TEST(NormalizeLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  double precision = 1e-5;
  LayerParameter layer_param;
  NormalizeLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i1 = 0; i1 < this->blob_bottom_->num(); ++i1) {
    Dtype normsqr_bottom = 0;
    Dtype normsqr_top = 0;
    for (int i2 = 0; i2 < this->blob_top_->channels(); ++i2) {
      for (int i3 = 0; i3 < this->blob_top_->height(); ++i3) {
        for (int i4 = 0; i4 < this->blob_top_->width(); ++i4) {
          normsqr_top += pow(this->blob_top_->data_at(i1,i2,i3,i4), 2);
          normsqr_bottom += pow(this->blob_bottom_->data_at(i1,i2,i3,i4), 2);
        }
      }
    }
    EXPECT_NEAR(normsqr_top, 1, precision);
    Dtype c = pow(normsqr_bottom, -0.5);
    for (int i2 = 0; i2 < this->blob_top_->channels(); ++i2) {
      for (int i3 = 0; i3 < this->blob_top_->height(); ++i3) {
        for (int i4 = 0; i4 < this->blob_top_->width(); ++i4) {
          EXPECT_NEAR(
            this->blob_top_->data_at(i1,i2,i3,i4), 
            this->blob_bottom_->data_at(i1,i2,i3,i4)*c, 
            precision);
        }
      }
    }
  }
}

TYPED_TEST(NormalizeLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  NormalizeLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
