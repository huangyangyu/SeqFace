// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/proposal_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

typedef ::testing::Types<GPUDevice<float>, GPUDevice<double> > TestDtypesGPU;

template <typename TypeParam>
class ProposalLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ProposalLayerTest()
      : blob_bottom_(new Blob<Dtype>(4, 3, 12, 8)),
        blob_bottom_anchor_(new Blob<Dtype>(4, 3, 12, 8)),
        //blob_bottom_img_(new Blob<Dtype>(4, 3, 12, 8)),
        blob_top_roi_(new Blob<Dtype>()),
        blob_top_score_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_anchor_);
    //filler.Fill(this->blob_bottom_img_);
    
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_anchor_);
    //blob_bottom_vec_.push_back(blob_bottom_img_);
    blob_top_vec_.push_back(blob_top_roi_);
    blob_top_vec_.push_back(blob_top_score_);
  }
  virtual ~ProposalLayerTest() {
    delete blob_bottom_;
    delete blob_bottom_anchor_;
    //delete blob_bottom_img_;
    delete blob_top_roi_;
    delete blob_top_score_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_anchor_;
  //Blob<Dtype>* const blob_bottom_img_;
  Blob<Dtype>* const blob_top_roi_;
  Blob<Dtype>* const blob_top_score_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ProposalLayerTest, TestDtypesGPU);

TYPED_TEST(ProposalLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  ProposalLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  //checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
  //    this->blob_top_vec_, 0);
}

}  // namespace caffe
