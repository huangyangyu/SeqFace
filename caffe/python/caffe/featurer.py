#!/usr/bin/env python
"""
Featurer is an image featurer specialization of Net.
"""

import numpy as np

import caffe


class Featurer(caffe.Net):
    """
    Featurer extends Net for image class prediction
    by scaling, center cropping, or oversampling.

    Parameters
    ----------
    image_dims : dimensions to scale input for cropping/sampling.
        Default is to scale to net input size for whole-image crop.
    mean, input_scale, raw_scale, channel_swap: params for
        preprocessing options.
    """
    def __init__(self, model_file, pretrained_file, image_dims=None,
                 mean=None, input_scale=None, raw_scale=None,
                 channel_swap=None):
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)

        # configure pre-processing
        in_ = self.inputs[0]
        self.transformer = caffe.io.Transformer(
            {in_: self.blobs[in_].data.shape})
        self.transformer.set_transpose(in_, (2, 0, 1))
        if mean is not None:
            mean_size = self.blobs[in_].data.shape[2:]
            if mean.ndim == 4:
                mean = mean[0]
            assert mean.ndim == 3
            mean =  np.transpose(mean, (1, 2, 0)) 
            mean = caffe.io.resize_image(mean, mean_size)
            mean =  np.transpose(mean, (2, 0, 1))
            self.transformer.set_mean(in_, mean)
        if input_scale is not None:
            self.transformer.set_input_scale(in_, input_scale)
        if raw_scale is not None:
            self.transformer.set_raw_scale(in_, raw_scale)
        if channel_swap is not None:
            self.transformer.set_channel_swap(in_, channel_swap)

        self.crop_dims = np.array(self.blobs[in_].data.shape[2:])
        if not image_dims:
            image_dims = self.crop_dims
        self.image_dims = image_dims

    def feature(self, inputs, feature_layer_names, oversample=True, norm=False):
        """
        feature extract
        """
        input_ = np.zeros((len(inputs),
                           self.image_dims[0],
                           self.image_dims[1],
                           inputs[0].shape[2]),
                          dtype=np.float32)
        for ix, in_ in enumerate(inputs):
            input_[ix] = caffe.io.resize_image(in_, self.image_dims)
        
        if oversample:
            # Generate center, corner, and mirrored crops.
            input_ = caffe.io.oversample(input_, self.crop_dims)
        else:
            center = np.array(self.image_dims) / 2.0
            crop = np.tile(center, (1, 2))[0] + np.concatenate([-self.crop_dims / 2.0, self.crop_dims / 2.0])
            crop = map(lambda x: int(x), crop)
            input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]

        # Feature
        caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]],
                            dtype=np.float32)
        for ix, in_ in enumerate(input_):
            caffe_in[ix] = self.transformer.preprocess(self.inputs[0], in_)
        out = self.forward_all(**{self.inputs[0]: caffe_in})
        
        features = list()
        for feature_layer_name in feature_layer_names:
            cand_layers = filter(lambda x: x.startswith(feature_layer_name), self.blobs.keys())
            if len(cand_layers) > 1:
                cand_layers = filter(lambda x: x == feature_layer_name, cand_layers)
            if len(cand_layers) == 1:
                feature_layer_name = cand_layers[0]
            else:
                return None
            
            feature = np.copy(self.blobs[feature_layer_name].data)
            size = feature.shape
            # For oversampling, average predictions across crops.
            if oversample:
                feature = feature.reshape(size[0], size[1])
                feature = feature.mean(0)
            else:
                feature = feature[0].reshape(size[1])

            if norm:
                feature = feature / np.linalg.norm(feature)

            features.append(feature)
        fusion_feature = np.concatenate(features, axis=0)
        fusion_feature = fusion_feature.astype(np.float32)
        return fusion_feature

