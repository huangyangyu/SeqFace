#!/usr/bin/env python
#coding: utf-8
#author: huangyangyu

import os
import sys
import glob
import logging
import traceback
from xml.etree import ElementTree


imageset_file = "./data/list.txt"
label_root_path = "/data/disk3/faster_rcnn_train/task/160310161831/Annotations/"
label_paths = map(lambda f: label_root_path + "/" + f + "/", \
                  filter(lambda f: os.path.isdir(label_root_path + "/" + f), \
		         os.listdir(label_root_path)))

log_file="./log/log.txt"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
	            filename=log_file,
	            filemode='a')


def obtain_imageset():
    # 获取图片数据集的列表
    with open(imageset_file, "w") as f:
	for label, path in enumerate(label_paths):
	    for fn in glob.glob(path + os.sep + "*.xml"):
	    	try:
		    root = ElementTree.parse(fn)
		    imageFile = root.find("filepath").text
		    assert os.path.exists(imageFile)

		    # default bound box
		    size = root.find("size")
		    width = size.find("width").text
		    height = size.find("height").text
		    depth = size.find("depth").text
		    assert int(depth) == 3

		    # labled bound box
		    for object in root.findall("object"):
		    	box = object.find("bndbox")
		    	xmin = box.find("xmin").text
		    	ymin = box.find("ymin").text
		    	xmax = box.find("xmax").text
		    	ymax = box.find("ymax").text
			assert int(xmin) <= int(xmax) <= int(width)
			assert int(ymin) <= int(ymax) <= int(height)
		    	f.write(imageFile + " " + str(xmin) + " " + str(ymin) + " " + \
			    	str(xmax) + " " + str(ymax) + " " + str(label) + "\n")
		except Exception as e:
		    logging.error(fn)
		    logging.error(imageFile)
		    logging.error(traceback.format_exc())


if __name__ == "__main__":
    # 入口函数
    obtain_imageset()

