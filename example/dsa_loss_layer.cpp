#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/dsa_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DSALossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    faceid_num_ = this->layer_param_.dsa_loss_param().faceid_num();
    DCHECK(faceid_num_ > 0);
    seqid_num_ = this->layer_param_.dsa_loss_param().seqid_num();
    DCHECK(seqid_num_ >= 0);
    N_ = faceid_num_ + seqid_num_;
    lambda_ = this->layer_param_.dsa_loss_param().lambda();
    DCHECK(lambda_ >= 0.0 && lambda_ <= 1.0);
    prob_ = Dtype(1.0) - this->layer_param_.dsa_loss_param().dropout_ratio();
    DCHECK(prob_ >= 0.0 && prob_ <= 1.0);
    ID_N_ = 1.0 * N_ * prob_;
    gamma_ = this->layer_param_.dsa_loss_param().gamma();
    DCHECK(gamma_ >= 1.0);
    alpha_ = this->layer_param_.dsa_loss_param().alpha();
    DCHECK(alpha_ >= 1.0);
    beta_ = this->layer_param_.dsa_loss_param().beta();
    DCHECK(beta_ >= 0.0);
    use_normalize_ = this->layer_param_.dsa_loss_param().use_normalize();
    if (use_normalize_)
    {
        scale_ = this->layer_param_.dsa_loss_param().scale();
        DCHECK(scale_ > 0.0);
    }
    center_iter_ = this->layer_param_.dsa_loss_param().center_iter();
    diff_iter_ = this->layer_param_.dsa_loss_param().diff_iter();
    iter_ = 0;
    has_ignore_label_ = this->layer_param_.loss_param().has_ignore_label();
    if (has_ignore_label_)
    {
        ignore_label_ = this->layer_param_.loss_param().ignore_label();
    }
    // Dimensions starting from "axis" are "flattened" into a single
    // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
    // and axis == 1, N inner products with dimension CHW are performed.
    const int axis = bottom[0]->CanonicalAxisIndex(this->layer_param_.dsa_loss_param().axis());
    K_ = bottom[0]->count(axis);
    // Check if we need to set up the weights
    if (this->blobs_.size() > 0)
    {
        LOG(INFO) << "Skipping parameter initialization";
    }
    else
    {
        // intialize the weights
        this->blobs_.resize(1);
        vector<int> center_shape(2);
        center_shape[0] = N_;
        center_shape[1] = K_;
        this->blobs_[0].reset(new Blob<Dtype>(center_shape));
        // fill the weights
        shared_ptr<Filler<Dtype> > center_filler(GetFiller<Dtype>(this->layer_param_.dsa_loss_param().center_filler()));
        center_filler->Fill(this->blobs_[0].get());
    }
    this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void DSALossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    M_ = bottom[0]->num();
    CHECK_EQ(bottom[1]->num(), M_);
    CHECK_EQ(bottom[1]->channels(), 1);
    CHECK_EQ(bottom[1]->height(), 1);
    CHECK_EQ(bottom[1]->width(), 1);
    LossLayer<Dtype>::Reshape(bottom, top);

    // id vec
    vector<int> id_vec_shape(1);
    id_vec_shape[0] = N_;
    id_vec_.Reshape(id_vec_shape);
    int* id_vec = id_vec_.mutable_cpu_data();
    for (int n = 0; n < N_; n++)
    {
        id_vec[n] = n;
    }
    // diff
    vector<int> diff_shape(1);
    diff_shape[0] = K_;
    diff_.Reshape(diff_shape);
    // square diff
    vector<int> square_diff_shape(2);
    square_diff_shape[0] = M_;
    square_diff_shape[1] = N_;
    square_diff_.Reshape(square_diff_shape);
    // variation sum
    vector<int> variation_sum_shape(1);
    variation_sum_shape[0] = K_;
    variation_sum_.Reshape(variation_sum_shape);
}

template <typename Dtype>
void DSALossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    ++iter_;

    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    Dtype* diff_data = diff_.mutable_cpu_data();
    Dtype* square_diff_data = square_diff_.mutable_cpu_data();

    if (use_normalize_)
    {
        // normalize center O(N*K)
        Dtype* norm_center = this->blobs_[0]->mutable_cpu_data();
        for (int n = 0; n < N_; n++)
        {
            Dtype sum_sqaure = caffe_cpu_dot(K_, norm_center + n * K_, norm_center + n * K_);
            caffe_scal(K_, Dtype(scale_/sqrt(sum_sqaure + 1e-7)), norm_center + n * K_);
        }

        // normalize bottom[0]
    }
    const Dtype* center = this->blobs_[0]->cpu_data();

    if (iter_ <= center_iter_ && iter_ <= diff_iter_)
    {
        return;
    }

    // id vec O(N)
    int* id_vec = id_vec_.mutable_cpu_data();
    std::random_shuffle(id_vec, id_vec + N_);

    // compute diff and square_diff O(M*ID_N*K)
    for (int m = 0; m < M_; m++)
    {
        // compute label_value
        int label_value = static_cast<int>(label[m]);
        if (has_ignore_label_ && label_value == ignore_label_)
        {
            continue;
        }
        if (label_value < 0)
        {
            if (seqid_num_ == 0)
            {
                continue;
            }
            label_value = faceid_num_ - 1 - label_value;
        }
        DCHECK_GE(label_value, 0);
        DCHECK_LT(label_value, N_);
        // compute square_diff
        for (int id_n = 0; id_n <= ID_N_; id_n++)
        {
            int n = (id_n == ID_N_ ? label_value : id_vec[id_n]);
            //Dtype* diff_m_n = diff_data + m * N_ * K_ + n * K_;
            Dtype* diff_m_n = diff_data;
            // D(m,n,:) = (X(m,:) - C(n,:)) / 2
            caffe_sub(K_, bottom_data + m * K_, center + n * K_, diff_m_n);
            caffe_scal(K_, Dtype(0.5), diff_m_n);
            // SD(m,n) = D(m,n,:) * D(m,n,:) = (x(m,:) - C(n,:)) ^ 2 / 4
            square_diff_data[m * N_ + n] = caffe_cpu_dot(K_, diff_m_n, diff_m_n);
        }
    }

    // loss O(M*ID_N)
    Dtype valid_count = Dtype(0.0);
    Dtype loss = Dtype(0.0);
    for (int m = 0; m < M_; m++)
    {
        int label_value = static_cast<int>(label[m]);
        if (has_ignore_label_ && label_value == ignore_label_)
        {
            continue;
        }
        if (label_value < 0)
        {
            if (seqid_num_ == 0)
            {
                continue;
            }
            label_value = faceid_num_ - 1 - label_value;
        }
        DCHECK_GE(label_value, 0);
        DCHECK_LT(label_value, N_);
        // center loss
        if (iter_ > center_iter_)
        {
            loss += lambda_ * square_diff_data[m * N_ + label_value];
        }
        // diff loss
        if (iter_ > diff_iter_)
        {
            for (int id_n = 0; id_n < ID_N_; id_n++)
            {
                int n = id_vec[id_n];
                if (n == label_value)
                {
                    continue;
                }
                if (label_value < faceid_num_ || n < faceid_num_)
                {
                    Dtype Jmn = alpha_ * square_diff_data[m * N_ + label_value] - square_diff_data[m * N_ + n] + beta_;
                    if (Jmn > Dtype(0.0))
                    {
                        loss += (Dtype(1.0) - lambda_) * Jmn / ID_N_;
                    }
                }
            }
        }
        valid_count += Dtype(1.0);
    }
    top[0]->mutable_cpu_data()[0] = loss / valid_count;
}

template <typename Dtype>
void DSALossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom)
{
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    const Dtype* center = this->blobs_[0]->cpu_data();
    Dtype* diff_data = diff_.mutable_cpu_data();
    const Dtype* square_diff_data = square_diff_.cpu_data();
    Dtype* variation_sum_data = variation_sum_.mutable_cpu_data();
    Dtype* center_diff = this->blobs_[0]->mutable_cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int* id_vec = id_vec_.cpu_data();
  
    // Gradient with respect to centers O(M*K+M*M)
    if (this->param_propagate_down_[0])
    {
        for (int m_ = 0; m_ < M_; m_++)
        {
            int n = static_cast<int>(label[m_]);
            if (has_ignore_label_ && n == ignore_label_)
            {
                continue;
            }
            if (n < 0)
            {
                if (seqid_num_ == 0)
                {
                    continue;
                }
                n = faceid_num_ - 1 - n;
            }
            DCHECK_GE(n, 0);
            DCHECK_LT(n, N_);
            int count = 0;
            caffe_set(K_, Dtype(0.0), variation_sum_data);
            for (int m = 0; m < M_; m++)
            {
                int label_value = static_cast<int>(label[m]);
                if (has_ignore_label_ && label_value == ignore_label_)
                {
                    continue;
                }
                if (label_value < 0)
                {
                    if (seqid_num_ == 0)
                    {
                        continue;
                    }
                    label_value = faceid_num_ - 1 - label_value;
                }
                DCHECK_GE(label_value, 0);
                DCHECK_LT(label_value, N_);
                if (label_value == n)
                {
                    // important: ensure that n is the first time to use
                    if (m < m_)
                        break;
                    //Dtype* diff_m_n = diff_data + m * N_ * K_ + label_value * K_;
                    Dtype* diff_m_n = diff_data;
                    caffe_sub(K_, bottom_data + m * K_, center + label_value * K_, diff_m_n);
                    caffe_scal(K_, Dtype(0.5), diff_m_n);
                    caffe_axpy(K_, -Dtype(1.0), diff_m_n, variation_sum_data);
                    ++count;
                }
            }
            caffe_axpy(K_, Dtype(1.0) / (count + Dtype(1.0)), variation_sum_data, center_diff + n * K_);
        }
    }

    if (iter_ <= center_iter_ && iter_ <= diff_iter_)
    {
        return;
    }

    // Gradient with respect to bottom data O(M*ID_N*K)
    if (propagate_down[0])
    {
        Dtype valid_count = Dtype(0.0);
        //caffe_set(M_ * K_, Dtype(0.0), bottom_diff);// maybe it can be deleted to make this loss adapt to parallel environment
        for (int m = 0; m < M_; m++)
        {
            int label_value = static_cast<int>(label[m]);
            if (has_ignore_label_ && label_value == ignore_label_)
            {
                continue;
            }
            if (label_value < 0)
            {
                if (seqid_num_ == 0)
                {
                    continue;
                }
                label_value = faceid_num_ - 1 - label_value;
            }
            DCHECK_GE(label_value, 0);
            DCHECK_LT(label_value, N_);
            // center loss
            if (iter_ > center_iter_)
            {
                //Dtype* diff_m_n = diff_data + m * N_ * K_ + label_value * K_;
                Dtype* diff_m_n = diff_data;
                caffe_sub(K_, bottom_data + m * K_, center + label_value * K_, diff_m_n);
                caffe_scal(K_, Dtype(0.5), diff_m_n);
                caffe_axpy(K_, lambda_, diff_m_n, bottom_diff + m * K_);
            }
            // diff loss
            if (iter_ > diff_iter_)
            {
                for (int id_n = 0; id_n < ID_N_; id_n++)
                {
                    int n = id_vec[id_n];
                    if (n == label_value)
                    {
                        continue;
                    }
                    if (label_value < faceid_num_ || n < faceid_num_)
                    {
                        Dtype Jmn = alpha_ * square_diff_data[m * N_ + label_value] - square_diff_data[m * N_ + n] + beta_;
                        if (Jmn > Dtype(0.0))
                        {
                            //Dtype* diff_m_n = diff_data + m * N_ * K_ + label_value * K_;
                            Dtype* diff_m_n = diff_data;
                            caffe_sub(K_, bottom_data + m * K_, center + label_value * K_, diff_m_n);
                            caffe_scal(K_, Dtype(0.5), diff_m_n);
                            caffe_axpy(K_, alpha_ * (Dtype(1.0) - lambda_) / ID_N_, diff_m_n, bottom_diff + m * K_);
                            
                            //Dtype* diff_m_n = diff_data + m * N_ * K_ + n * K_;
                            //Dtype* diff_m_n = diff_data;
                            caffe_sub(K_, bottom_data + m * K_, center + n * K_, diff_m_n);
                            caffe_scal(K_, Dtype(0.5), diff_m_n);
                            caffe_axpy(K_, -(Dtype(1.0) - lambda_) / ID_N_, diff_m_n, bottom_diff + m * K_);
                        }
                    }
                }
            }
            valid_count += Dtype(1.0);
        }
        caffe_scal(M_ * K_, top[0]->cpu_diff()[0] / valid_count, bottom_diff);
    }
    if (propagate_down[1])
    {
        LOG(FATAL) << this->type()
                   << " Layer cannot backpropagate to label inputs.";
    }
}

#ifdef CPU_ONLY
STUB_GPU(DSALossLayer);
#endif

INSTANTIATE_CLASS(DSALossLayer);
REGISTER_LAYER_CLASS(DSALoss);

}  // namespace caffe
