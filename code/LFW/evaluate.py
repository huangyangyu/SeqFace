#!/usr/bin/env python
#coding: utf-8
#author: huangyangyu

layer_num = 27
#layer_num = 64

import os
import sys
import gflags
import cPickle
import numpy as np

root_dir = os.path.dirname(os.path.abspath("__file__")) + "/../../"
sys.path.append(root_dir + "code/")

if not gflags.FLAGS.has_key("model_dir"):
    gflags.DEFINE_string("model_dir", root_dir + "model/ResNet-%d/" % layer_num, "set model dir")
if not gflags.FLAGS.has_key("feature_layer_names"):
    gflags.DEFINE_string("feature_layer_names", "['fc5']", "set feature layer names")
if not gflags.FLAGS.has_key("device_id"):
    gflags.DEFINE_integer("device_id", 0, "set device id")
if not gflags.FLAGS.has_key("ratio"):
    gflags.DEFINE_float("ratio", -1.0, "set image ratio")
if not gflags.FLAGS.has_key("scale"):
    gflags.DEFINE_float("scale", 1.1, "set image scale")
if not gflags.FLAGS.has_key("resize_height"):
    gflags.DEFINE_integer("resize_height", 144, "set image height")
if not gflags.FLAGS.has_key("resize_width"):
    gflags.DEFINE_integer("resize_width", 144, "set image width")
if not gflags.FLAGS.has_key("raw_scale"):
    gflags.DEFINE_float("raw_scale", 255.0, "set raw scale")
if not gflags.FLAGS.has_key("input_scale"):
    gflags.DEFINE_float("input_scale", 0.0078125, "set raw scale")
if not gflags.FLAGS.has_key("gray"):
    gflags.DEFINE_boolean("gray", False, "set gray")
if not gflags.FLAGS.has_key("oversample"):
    gflags.DEFINE_boolean("oversample", False, "set oversample")

from featurer import Featurer


def cos_sim(v1, v2):
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    dist = np.linalg.norm(np.array(v1)-np.array(v2))
    cos = 1 - dist * dist / 2
    return cos


def test():
    image_dir = root_dir + "data/LFW/"

    # pairs
    image_files = set()
    pairs = list()
    for k, line in enumerate(open(image_dir + "pairs.txt")):
        item = line.strip().split()
        item[0] = image_dir + "images/" + item[0]
        item[1] = image_dir + "images/" + item[1]
        assert len(item) == 3
        pairs.append(tuple(item))
        image_files.add(item[0])
        image_files.add(item[1])

    # features
    feature_file = image_dir + "feature_%d.pkl" % layer_num
    if not os.path.exists(feature_file):
        gflags.FLAGS(sys.argv)
        model_dir = gflags.FLAGS.model_dir + "/"
        featurer = Featurer(deploy_prototxt=model_dir + "deploy.prototxt", \
                            model_file=model_dir + "train.caffemodel", \
                            mean_file=model_dir + "mean.binaryproto", \
                            ratio_file=model_dir + "ratio.txt", \
                            label_file=model_dir + "label.txt", \
                            device_id=gflags.FLAGS.device_id, \
                            ratio=gflags.FLAGS.ratio, \
                            scale=gflags.FLAGS.scale, \
                            resize_height=gflags.FLAGS.resize_height, \
                            resize_width=gflags.FLAGS.resize_width, \
                            raw_scale=gflags.FLAGS.raw_scale, \
                            input_scale=gflags.FLAGS.input_scale, \
                            gray=gflags.FLAGS.gray, \
                            oversample=gflags.FLAGS.oversample, \
                            feature_layer_names=eval(gflags.FLAGS.feature_layer_names))
        features = dict()
        for k, image_file in enumerate(image_files):
            if not features.has_key(image_file):
                features[image_file.replace(root_dir, "")] = featurer.test(image_file=image_file)
            print "processed:", k
            sys.stdout.flush()
        cPickle.dump(features, open(feature_file, "wb"))
    else:
        features = cPickle.load(open(feature_file, "rb"))

    # sims
    sims = list()
    for pair in pairs:
        image_file1, image_file2, tag = pair[:3]
        # person1
        feature1 = features[image_file1.replace(root_dir, "")]
        # person2
        feature2 = features[image_file2.replace(root_dir, "")]
        # sim
        sim = cos_sim(feature1, feature2)
        sims.append((sim, int(tag), image_file1, image_file2))
    sims = sorted(sims, key=lambda item: item[0])

    # roc
    tn = 0
    fn = 0
    tp = len(filter(lambda item: item[1]==1, sims))
    fp = len(filter(lambda item: item[1]==0, sims))
    best_accuracy = 0.0
    best_thred = 0.0
    with open(image_dir + "roc_%d.txt" % layer_num, "wb") as f:
        for k, sim in enumerate(sims):
            thred, tag, image_file1, image_file2 = sim
            if tag == 0:
                tn += 1
                fp -= 1
            else:
                fn += 1
                tp -= 1
            tpr = 1.0 * tp / max(tp + fn, 1)
            fnr = 1.0 * fn / max(tp + fn, 1)
            tnr = 1.0 * tn / max(tn + fp, 1)
            fpr = 1.0 * fp / max(tn + fp, 1)
            accuracy = 1.0 * (tp + tn) / (tp + fp + tn + fn)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_thred = thred
            f.write("%.6f %.6f\n" % (tpr, fpr))
    print "best:", len(pairs), best_thred, best_accuracy


if __name__ == "__main__":
    test()
