#!/usr/bin/env python
#coding: utf-8
#author: huangyangyu

caffe_dir = "/root/image_server/lib/caffe/python/"

import os
import sys
sys.path.insert(0, caffe_dir)
import util
import gflags
import numpy as np
import caffe


if not gflags.FLAGS.has_key("model_dir"):
    gflags.DEFINE_string("model_dir", "@model_path", "set model dir")
if not gflags.FLAGS.has_key("feature_layer_names"):
    gflags.DEFINE_string("feature_layer_names", "['@feature_layer_name']", "set feature layer names")
if not gflags.FLAGS.has_key("device_id"):
    gflags.DEFINE_integer("device_id", 0, "set device id")
if not gflags.FLAGS.has_key("ratio"):
    gflags.DEFINE_float("ratio", -1.0, "set image ratio")
if not gflags.FLAGS.has_key("scale"):
    gflags.DEFINE_float("scale", 1.1, "set image scale")
if not gflags.FLAGS.has_key("resize_height"):
    gflags.DEFINE_integer("resize_height", 256, "set image size")
if not gflags.FLAGS.has_key("resize_width"):
    gflags.DEFINE_integer("resize_width", 256, "set image size")
if not gflags.FLAGS.has_key("raw_scale"):
    gflags.DEFINE_float("raw_scale", 255.0, "set raw scale")
if not gflags.FLAGS.has_key("input_scale"):
    gflags.DEFINE_float("input_scale", 1.0, "set input scale")
if not gflags.FLAGS.has_key("gray"):
    gflags.DEFINE_boolean("gray", False, "set gray")
if not gflags.FLAGS.has_key("oversample"):
    gflags.DEFINE_boolean("oversample", False, "set oversample")


class Featurer():
    def __init__(self, deploy_prototxt, model_file, mean_file=None, ratio_file=None, label_file=None, device_id=-1, score_thred=0.0, top_k=1, ratio=-1.0, scale=1.1, resize_height=256, resize_width=256, raw_scale=255.0, input_scale=1.0, gray=False, oversample=False, feature_layer_names=None):
        self.ratio = ratio
        self.scale = scale
        self.gray = gray
        self.oversample = oversample
        self.feature_layer_names = feature_layer_names

        if device_id < 0:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(device_id)

        mean = None
        if mean_file and os.path.exists(mean_file):
            mean_suffix = os.path.splitext(mean_file)[1]
            if mean_suffix == ".binaryproto":
                binary_data = open(mean_file, "rb").read()
                proto_data = caffe.io.caffe_pb2.BlobProto.FromString(binary_data)
                mean = caffe.io.blobproto_to_array(proto_data)
            elif mean_suffix == ".npy":
                mean = np.load(mean_file)
        
        self.featurer = caffe.Featurer(deploy_prototxt, model_file, \
                                       image_dims=(resize_height, resize_width), mean=mean, \
                                       raw_scale=raw_scale, input_scale=input_scale, \
                                       channel_swap=(2, 1, 0) if not self.gray else None)
        

    def test(self, image_file, box=None, feature_layer_names=None, score_thred=0.0, top_k=1, ratio=None, scale=None, oversample=None, timeout=2):
        if ratio is None:
            ratio = self.ratio
        if scale is None:
            scale = self.scale
        if oversample is None:
            oversample = self.oversample
        if feature_layer_names is None:
            feature_layer_names = self.feature_layer_names

        input = caffe.io.load_image(image_file, not self.gray)
        if input is None:
            raise Exception("Unreadable image file: " + image_file)
        
        input, box = util.align_box(input, box)
        height, width, channels = input.shape
        if box is not None:
            if box["w"] <= 1.0 and box["h"] <= 1.0:
                box["x"] = int(round(box["x"] * width))
                box["y"] = int(round(box["y"] * height))
                box["w"] = int(round(box["w"] * width))
                box["h"] = int(round(box["h"] * height))
            else:
                box["x"] = int(round(box["x"]))
                box["y"] = int(round(box["y"]))
                box["w"] = int(round(box["w"]))
                box["h"] = int(round(box["h"]))
            box = util.fix_box(box=box, width=width, height=height, ratio=ratio, scale=scale)
            if box["w"] == 0 or box["h"] == 0:
                raise Exception("Invalid box shape of image file: " + image_file)
            input = input[box["y"]: box["y"]+box["h"], box["x"]: box["x"]+box["w"]]
        if input.shape[0] < 8 or input.shape[1] < 8 or \
           8.0 * input.shape[0] < input.shape[1] or 8.0 * input.shape[1] < input.shape[0]:
            raise Exception("Inappropriate box shape of image file: " + image_file)
        input = [input]

        feature = self.featurer.feature(inputs=input, feature_layer_names=feature_layer_names, oversample=oversample)
        return feature

