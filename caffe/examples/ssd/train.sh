#!/usr/bin/env bash

cd /root/test/caffe
./build/tools/caffe train \
--solver="/home/user/steven/sample_gen/flickr_par_trainval/trainer/solver.prototxt" \
--weights="/home/user/steven/sample_gen/flickr_par_trainval/trainer/pretrain.caffemodel" \
--gpu 1

