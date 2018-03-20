#!/usr/bin/env python
#coding: utf-8
#author: huangyangyu

import cv2
import math
import copy
import heapq
import tempfile
import numpy as np


def fix_box(box, width, height, ratio, scale=1.0):
    box = copy.deepcopy(box)
    w = box["w"]
    h = box["h"]
    x = box["x"] + w / 2
    y = box["y"] + h / 2
    mw = 2 * min(x, width - x)
    mh = 2 * min(y, height - y)
    w = max(1, min(int(w * scale), mw))
    h = max(1, min(int(h * scale), mh))
    if ratio > 0:
        if 1.0 * w / h > ratio:
            h = int(w / ratio)
            h = min(h, mh)
            w = int(h * ratio)
        else:
            w = int(h * ratio)
            w = min(w, mw)
            h = int(w / ratio)
    box["x"] = x - w / 2
    box["y"] = y - h / 2
    box["w"] = w
    box["h"] = h
    return box


def rotate_point(p, c, rad):
    x = p["x"] - c["x"]
    y = p["y"] - c["y"]
    _p = dict()
    _p["x"] = +x * np.cos(rad) + y * np.sin(rad) + c["x"]
    _p["y"] = -x * np.sin(rad) + y * np.cos(rad) + c["y"]
    return _p


def align_box(input, box, method=1):
    if box is None or not box.has_key("landmark"):
        return input, box

    # scale input
    height, width, channels = input.shape
    landmark = copy.deepcopy(box["landmark"])
    for key in landmark.keys():
        landmark[key]["x"] = int(landmark[key]["x"] * width)
        landmark[key]["y"] = int(landmark[key]["y"] * height)
    # eye center and mouse center point
    eye_center = dict()
    eye_center["x"] = (landmark["eye_left"]["x"] + landmark["eye_right"]["x"]) / 2
    eye_center["y"] = (landmark["eye_left"]["y"] + landmark["eye_right"]["y"]) / 2
    mouse_center = dict()
    mouse_center["x"] = (landmark["mouse_left"]["x"] + landmark["mouse_right"]["x"]) / 2
    mouse_center["y"] = (landmark["mouse_left"]["y"] + landmark["mouse_right"]["y"]) / 2
    nose = landmark["nose"]
    # rotation
    rad_tan = 1.0 * (landmark["eye_right"]["y"] - landmark["eye_left"]["y"]) / (landmark["eye_right"]["x"] - landmark["eye_left"]["x"])
    rad = math.atan(rad_tan)
    deg = np.rad2deg(rad)
    width, height = (int((abs(np.sin(rad)*height) + abs(np.cos(rad)*width))), int((abs(np.cos(rad)*height) + abs(np.sin(rad)*width))))
    transformMat = cv2.getRotationMatrix2D((eye_center["x"], eye_center["y"]), deg, 1.0)
    input = cv2.warpAffine(input, transformMat, (width, height))
    if len(input.shape) == 2:
        input = input.reshape((height, width, 1))
    # align box
    if method == 1:
        #d = np.linalg.norm((mouse_center["x"] - eye_center["x"], mouse_center["y"] - eye_center["y"]))
        r_mouse_center = rotate_point(mouse_center, eye_center, rad)
        r_eye_center = rotate_point(eye_center, eye_center, rad)
        d = r_mouse_center["y"] - r_eye_center["y"] + 1
        s = 3 * d
        dx = int(s / 2)
        dy = int(s / 3)
    elif method == 2:
        d = np.linalg.norm((landmark["eye_right"]["x"] - landmark["eye_left"]["x"], landmark["eye_right"]["y"] - landmark["eye_left"]["y"]))
        s = 3 * d
        dx = int(s / 2)
        dy = int(s * (3 - math.sqrt(5)) / 2)
    x0 = eye_center["x"] - dx
    x1 = eye_center["x"] + (s - dx) - 1
    x0, x1 = (max(int(x0), 0), min(int(x1), width-1))
    y0 = eye_center["y"] - dy
    y1 = eye_center["y"] + (s - dy) - 1
    y0, y1 = (max(int(y0), 0), min(int(y1), height-1))
    box = dict()
    box["x"] = x0
    box["y"] = y0
    box["w"] = x1 - x0 + 1
    box["h"] = y1 - y0 + 1

    return input, box

