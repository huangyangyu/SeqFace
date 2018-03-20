#!/usr/bin/env python
#coding: utf-8
#author: huangyangyu

import os
import sys
sys.path.append("/root/image_server/bin/core/test/")##
sys.path.append("/root/test/caffe/python/")##
#sys.path.append("/root/image_server/lib/caffe/python/")##
from tester import Tester
import caffe
import cv2
import numpy as np
import time
import tool
import threading
import gflags
import json
import collections
import traceback

if not gflags.FLAGS.has_key("model_dir"):
    gflags.DEFINE_string("model_dir", "/data/image_server/task/detect/car/ssd_test/trainer_car/", "set model dir")
if not gflags.FLAGS.has_key("data_dir"):
    gflags.DEFINE_string("data_dir", "/data/image_server/task/detect/car/ssd_test/converter_car/", "set data dir")
if not gflags.FLAGS.has_key("device_id"):
    gflags.DEFINE_integer("device_id", 0, "set device id")
if not gflags.FLAGS.has_key("score_thred"):
    gflags.DEFINE_string("score_thred", "0.5", "set score thred")
if not gflags.FLAGS.has_key("top_k"):
    gflags.DEFINE_integer("top_k", -1, "set topk")
if not gflags.FLAGS.has_key("ratio"):#useless
    gflags.DEFINE_float("ratio", -1.0, "set ratio")
if not gflags.FLAGS.has_key("image_size"):
    gflags.DEFINE_integer("image_size", 300, "set image size")
if not gflags.FLAGS.has_key("raw_scale"):
    gflags.DEFINE_float("raw_scale", 255.0, "set raw scale")
if not gflags.FLAGS.has_key("gray"):
    gflags.DEFINE_boolean("gray", False, "set gray")
if not gflags.FLAGS.has_key("mean_value"):
    gflags.DEFINE_string("mean_value", "(0.0, 0.0, 0.0)", "set mean value")# r, g, b


class Detector(Tester):
    """
    检测
    """
    def __init__(self, deploy_prototxt, model_file, mean_file=None, mean_value=None, ratio_file=None, label_file=None, device_id=-1, score_thred=0.0, top_k=-1, ratio=-1.0, image_size=256, raw_scale=255.0, gray=False):
        """
        初始化
        """
        Tester.__init__(self)

        self.score_thred = score_thred
        self.top_k = top_k
        self.ratio = ratio
        self.gray = gray

        self.ratios = None
        if ratio_file is not None and os.path.exists(ratio_file):
            self.ratios = dict()
            for i, line in enumerate(open(ratio_file)):
                items = line.strip(" \0\t\r\n").split(";")
                self.ratios[i] = float(items[1])

        self.labels = None
        if label_file is not None:
            self.labels = open(label_file).readlines()
            self.labels = filter(lambda x: len(x.split(";")) == 2, self.labels)
            self.labels = map(lambda x: x.strip().decode("utf-8"), self.labels)

        if device_id < 0:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(device_id)
        
        mean = None
        # mean: num, channels, height, width
        if mean_file and os.path.exists(mean_file):
            mean_suffix = os.path.splitext(mean_file)[1]
            if mean_suffix == ".binaryproto":
                binary_data = open(mean_file, "rb").read()
                proto_data = caffe.io.caffe_pb2.BlobProto.FromString(binary_data)
                mean = caffe.io.blobproto_to_array(proto_data)
            elif mean_suffix == ".npy":
                mean = np.load(mean_file)
        elif mean_value is not None:
            mean = np.tile(np.array(mean_value), (1, image_size, image_size, 1))
            mean = mean.transpose((0, 3, 1, 2))
        
        if mean is not None and mean.ndim == 4:
            mean = mean[0]
        assert mean is None or mean.ndim == 3

        self.detector = caffe.Detector(deploy_prototxt, model_file, \
                                       image_dims=(image_size, image_size), mean=mean, \
                                       raw_scale=raw_scale, channel_swap=(2, 1, 0) if not self.gray else None)

        # thread lock
        self.mutex = threading.Lock()

        # init over
        self.is_init = True

        # warm up
        #self.test(image_file="./data/warm_up.jpg")
        

    def type(self):
        """
        类型
        """
        return "Detector"


    @staticmethod
    def update():
        """
        重载算法库
        """
        reload(caffe)


    def test(self, image_file, box=None, score_thred=None, top_k=None, ratio=None, need_fix_box=False, timeout=2):
        """
        测试
        """
        #s = time.time()

        objects = []
        if not self.is_init:
            raise Exception("Not initialized tester")

        if score_thred is None:
            score_thred = self.score_thred
        if top_k is None:
            top_k = self.top_k
        if ratio is None:
            ratio = self.ratio

        input = caffe.io.load_image(image_file, not self.gray)
        if input is None:
            raise Exception("Unreadable image file: " + image_file)

        input, box = tool.align_box(input, box)
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
            box = tool.fix_box(box=box, width=width, height=height, ratio=ratio, scale=1.1)
            if box["w"] == 0 or box["h"] == 0:
                raise Exception("Invalid box shape of image file: " + image_file)
            input = input[box["y"]: box["y"]+box["h"], box["x"]: box["x"]+box["w"]]
        if input.shape[0] < 8 or input.shape[1] < 8 or \
           8.0 * input.shape[0] < input.shape[1] or 8.0 * input.shape[1] < input.shape[0]:
            raise Exception("Inappropriate box shape of image file: " + image_file)
        input = [input]

        detections = None
        if self.mutex.acquire(int(timeout)):
            detections = self.detector.detect(input)
            self.mutex.release()

        for detection in detections:
            object = dict()
            id = int(detection[1]) - 1
            object["classId"] = id
            if self.labels is not None and 0 <= id < len(self.labels):
                object["name"] = self.labels[id]
            object["score"] = round(detection[2], 4)
            box = dict()
            detection[3:] = map(lambda det: min(max(det, 0.0), 1.0), detection[3:])
            box["x"] = round(detection[3], 4)
            box["y"] = round(detection[4], 4)
            box["w"] = round(detection[5] - detection[3], 4)
            box["h"] = round(detection[6] - detection[4], 4)
            object["box"] = box
            objects.append(object)
        
        # top k
        objects = tool.top_k(objects, top_k, score_thred)

        #print time.time() - s
        #sys.stdout.flush()
        return objects


if __name__ == '__main__':
    """
    入口函数
    """
    gflags.FLAGS(sys.argv)
    model_dir = gflags.FLAGS.model_dir + "/"
    detector = Detector(deploy_prototxt=model_dir + "deploy.prototxt", \
                        model_file=model_dir + "train.caffemodel", \
                        mean_file=model_dir + "mean.binaryproto", \
                        mean_value = eval(gflags.FLAGS.mean_value), \
                        ratio_file=model_dir + "ratio.txt", \
                        label_file=model_dir + "label.txt", \
                        device_id=gflags.FLAGS.device_id, \
                        score_thred=eval(gflags.FLAGS.score_thred), \
                        top_k=gflags.FLAGS.top_k, \
                        ratio=gflags.FLAGS.ratio, \
                        image_size=gflags.FLAGS.image_size, \
                        raw_scale=gflags.FLAGS.raw_scale, \
                        gray=gflags.FLAGS.gray)
    print detector.test(image_file="/root/test/caffe/build/examples/ssd/A.jpg")

