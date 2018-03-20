#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Compute_distance_data_gpu(int nthreads, const int K, 
                                          const Dtype* bottom, const Dtype* label, 
                                          const Dtype* center, Dtype* distance, 
                                          const bool has_ignore_label_, const int ignore_label_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int m = index / K;
    int k = index % K;
    const int label_value = static_cast<int>(label[m]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      distance[index] = 0;
    } else {
      // distance(i) = x(i) - c_{y(i)}
      distance[index] = bottom[index] - center[label_value * K + k];
    }
  }
}

template <typename Dtype>
__global__ void Compute_center_diff_gpu(int nthreads, const int M, const int K, 
                                        const Dtype* label, const Dtype* distance, 
                                        Dtype* variation_sum, Dtype* center_diff, 
                                        const bool has_ignore_label_, const int ignore_label_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int count = 0;
    for (int m = 0; m < M; m++) {
      const int label_value = static_cast<int>(label[m]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        // pass
      } else {
        if (label_value == index) {
          count++;
          for (int k = 0; k < K; k++) {
            variation_sum[index * K + k] -= distance[m * K + k];
          }
        }
      }
    }
    for (int k = 0; k < K; k++) {
      center_diff[index * K + k] = variation_sum[index * K + k] /(count + (Dtype)1.);
      //center_diff[index * K + k] += variation_sum[index * K + k] /(count + (Dtype)1.);
    }
  }
}

/*
template <typename Dtype>
__global__ void Compute_variation_sum_gpu(int nthreads, const int K, 
                                          const Dtype* label, const Dtype* distance, 
                                          Dtype* variation_sum, int* count, 
                                          const bool has_ignore_label_, const int ignore_label_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int m = index / K;
    int k = index % K;
    const int label_value = static_cast<int>(label[m]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      // pass
    } else {
      variation_sum[label_value * K + k] -= distance[m * K + k];
      if (k == 0)
      {
        ++count[label_value];
      }
    }
  }
}

template <typename Dtype>
__global__ void Compute_center_diff_gpu(int nthreads, const int K, 
                                        const Dtype* label, Dtype* variation_sum, 
                                        int* count, Dtype* center_diff, 
                                        const bool has_ignore_label_, const int ignore_label_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int m = index / K;
    int k = index % K;
    const int n = static_cast<int>(label[m]);
    if (has_ignore_label_ && n == ignore_label_) {
      // pass
    } else {
      if (count[n] > 0)
      {
        center_diff[n * K + k] += variation_sum[n * K + k] / (count[n] + (Dtype)1.);
        count[n] = 0;
      }
    }
  }
}
*/

template <typename Dtype>
void CenterLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  int nthreads = M_ * K_;
  Compute_distance_data_gpu<Dtype> <<< CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, bottom[0]->gpu_data(),
                                bottom[1]->gpu_data(),
                                this->blobs_[0]->gpu_data(),
                                distance_.mutable_gpu_data(),
                                has_ignore_label_,
                                ignore_label_);
  Dtype dot;
  caffe_gpu_dot(M_ * K_, distance_.gpu_data(), distance_.gpu_data(), &dot);
  Dtype valid_count = Dtype(0.0);
  const Dtype* label = bottom[1]->cpu_data();
  for (int i = 0; i < M_; i++) {
    const int label_value = static_cast<int>(label[i]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      continue;
    }
    valid_count += Dtype(1.0);
  }
  Dtype loss = dot / valid_count / Dtype(2);
  //Dtype loss = dot / M_ / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                          const vector<bool>& propagate_down,
                                          const vector<Blob<Dtype>*>& bottom) {
  caffe_gpu_set(N_ * K_, (Dtype)0., variation_sum_.mutable_gpu_data());
  int nthreads = N_;
  Compute_center_diff_gpu<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, M_, K_,
                                  bottom[1]->gpu_data(),
                                  distance_.gpu_data(),
                                  variation_sum_.mutable_cpu_data(),
                                  this->blobs_[0]->mutable_gpu_diff(),
                                  has_ignore_label_,
                                  ignore_label_);
  /*
  caffe_gpu_set(N_, 0 , count_.mutable_gpu_data());
  int nthreads = M_ * K_;
  Compute_variation_sum_gpu<Dtype> <<< CAFFE_GET_BLOCKS(nthreads),
    CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, bottom[1]->gpu_data(),
                              distance_.gpu_data(),
                              variation_sum_.mutable_gpu_data(),
                              count_.mutable_gpu_data(),
                              has_ignore_label_,
                              ignore_label_);
  Compute_center_diff_gpu<Dtype> <<< CAFFE_GET_BLOCKS(nthreads),
    CAFFE_CUDA_NUM_THREADS>>>(nthreads, K_, bottom[1]->gpu_data(),
                              variation_sum_.mutable_gpu_data(),
                              count_.mutable_gpu_data(),
                              this->blobs_[0]->mutable_gpu_diff(),
                              has_ignore_label_,
                              ignore_label_);
  */
  if (propagate_down[0]) {
    Dtype valid_count = Dtype(0.0);
    const Dtype* label = bottom[1]->cpu_data();
    for (int i = 0; i < M_; i++) {
      const int label_value = static_cast<int>(label[i]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      valid_count += Dtype(1.0);
    }
    caffe_gpu_scale(M_ * K_, top[0]->cpu_diff()[0] / valid_count, 
                             distance_.gpu_data(), bottom[0]->mutable_gpu_diff());
    //caffe_gpu_scale(M_ * K_, top[0]->cpu_diff()[0] / M_, 
    //                         distance_.gpu_data(), bottom[0]->mutable_gpu_diff());
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CenterLossLayer);

}  // namespace caffe
