#!/usr/bin/env bash
#author: huangyangyu

mode=1
#mode=2

dataset=LFW
#dataset=YTF

if [ $# -eq 2 ]; then
    mode=$1
    dataset=$2
fi
echo "mode: ${mode}, dataset: ${dataset}"

#step 1: git clone https://github.com/ydwen/caffe-face.git
#Done

#step 2: compile caffe
if [ ! -d caffe/build/ ]; then
    echo "compiling caffe"
    cd caffe
    mkdir build
    cd build
    cmake ..
    make -j32
    make pycaffe
    cd ../..
fi

#step 3: download model and testing dataset, then unzip them
if [ $mode -eq 1 ]; then
    # feature
    cd data/${dataset}
    feature_file=feature_${dataset}.tar.gz
    if [ ! -f "${feature_file}" ]; then
        echo "downloading feature file"
        wget http://imgserver.yunshitu.cn/extra/user/huangyangyu/${feature_file}
        tar -zxf ${feature_file}
    fi
    cd -
else
    # model
    cd model
    model_file=ResNet-27.tar.gz
    if [ ! -f "${model_file}" ]; then
        echo "downloading model file"
        wget http://imgserver.yunshitu.cn/extra/user/huangyangyu/${model_file}
        tar -zxf ${model_file}
    fi
    cd -
    # dataset
    cd data/${dataset}
    data_file=${dataset}.tar.gz
    if [ ! -f "$data_file" ]; then
        echo "downloading data file"
        wget http://imgserver.yunshitu.cn/extra/user/huangyangyu/${data_file}
        tar -zxf ${data_file}
    fi
    cd -
fi

#step 4: run evaluate.py in LFW or YTF directory
cd code/${dataset}
python evaluate.py
cd -

