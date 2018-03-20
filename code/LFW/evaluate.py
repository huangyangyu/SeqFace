#!/usr/bin/env python
#coding: utf-8
#author: huangyangyu

root_dir = "/data/image_server/user/huangyangyu/seqface/SeqFace/"
layer_num = 27
#layer_num = 64

import os
import sys
sys.path.append(root_dir + "code/")
import gflags
import cPickle
import numpy as np

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
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(v1, v2):
        dot_product += a * b
        normA += a**2
        normB += b**2
    if normA != 0.0 and normB != 0.0:
        cos = dot_product / ((normA*normB)**0.5)
    else:
        cos = 1.0
    sim = 0.5 + 0.5 * cos
    return sim


def norml2_sim(v1, v2):
    normA = 0.0
    normB = 0.0
    for a, b in zip(v1, v2):
        normA += a ** 2
        normB += b ** 2
    normA **= 0.5
    normB **= 0.5
    diff = 0.0
    for a, b in zip(v1, v2):
        a /= normA
        b /= normB
        diff += (a - b) ** 2
    if normA != 0.0 and normB != 0.0:
        diff **= 0.5
    else:
        diff = 1.0
    sim = 1.0 - 0.5 * diff
    return sim


def test():
    image_dir = root_dir + "data/LFW/"

    image_files = set()
    pairs = list()
    for k, line in enumerate(open(image_dir + "pairs.txt")):
        item = line.strip().split()
        item[0] = image_dir + "images/" + item[0]
        item[1] = image_dir + "images/" + item[1]
        assert len(item) == 3
        if os.path.exists(item[0]) and os.path.exists(item[1]):
            pairs.append(tuple(item))
            image_files.add(item[0])
            image_files.add(item[1])
        #else:
        #    print item[0]
        #    print item[1]

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
                features[image_file] = featurer.test(image_file=image_file)
            print "processed:", k
            sys.stdout.flush()
        cPickle.dump(features, open(feature_file, "wb"))
    else:
        features = cPickle.load(open(feature_file, "rb"))

    sims = list()
    threds = list()
    for pair in pairs:
        image_file1, image_file2, tag = pair[:3]
        # person1
        feature1 = features[image_file1]
        # person2
        feature2 = features[image_file2]
        # sim
        #sim = cos_sim(feature1, feature2)
        sim = norml2_sim(feature1, feature2)
        sims.append((sim, int(tag), image_file1, image_file2))
        threds.append(sim)

    best_accuracy = 0.0
    best_thred = 0.0
    with open(image_dir + "roc_%d.txt" % layer_num, "wb") as f:
        for thred in sorted(threds):
            tp = 0
            fn = 0
            tn = 0
            fp = 0
            for sim, tag, image_file1, image_file2 in sims:
                if tag == 1:
                    if sim >= thred:
                        tp += 1
                    else:
                        fn += 1
                        #print "fp", image_file1, image_file2
                if tag == 0:
                    if sim < thred:
                        tn += 1
                    else:
                        fp += 1
                        #print "fn", image_file1, image_file2
            tpr = 1.0 * tp / max(tp + fn, 1)
            fnr = 1.0 * fn / max(tp + fn, 1)
            tnr = 1.0 * tn / max(tn + fp, 1)
            fpr = 1.0 * fp / max(tn + fp, 1)
            accuracy = 1.0 * (tp + tn) / (tp + fp + tn + fn)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_thred = thred
            f.write("%.6f %.6f\n" % (tpr, fpr))
            #print thred, (tp + fp + tn + fn), tpr, tnr, accuracy
        print "best:", len(pairs), best_thred, best_accuracy


if __name__ == "__main__":
    test()

