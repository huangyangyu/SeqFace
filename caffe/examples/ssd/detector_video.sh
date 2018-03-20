#!/usr/bin/env bash
#coding: utf-8

caffe_root=/root/test/caffe/
iterations=65536
device_id=2
#model_file=/root/test/caffe/models/VGGNet/VOC0712Plus/SSD_300x300_ft/test.prototxt
#weight_file=/root/test/caffe/models/VGGNet/VOC0712Plus/SSD_300x300_ft/train.caffemodel
model_file=/data/image_server/task/detect/car/ssd_test/trainer_car/deploy_video.prototxt
weight_file=/data/image_server/task/detect/car/ssd_test/trainer_car/train.caffemodel

${caffe_root}/build/tools/caffe test \
    --model=${model_file} \
    --weights=${weight_file} \
    --iterations=${iterations} \
    --gpu=${device_id}

