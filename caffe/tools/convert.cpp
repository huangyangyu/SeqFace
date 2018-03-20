#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include <map>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>

#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

#include "gflags/gflags.h"
#include "glog/logging.h"

DEFINE_string(data_dir, "", "Db root dir where 'train' and 'test' dataset are stored. ");
DEFINE_string(train_val_test_ratios, "[0.8,0.1,0.1]", "Decide train/val/test set ratios");
DEFINE_bool(shuffle, true, "Decide whether shuffle data.");

using boost::scoped_ptr;
using boost::shared_ptr;
using boost::property_tree::ptree;
using namespace caffe;

struct Bndbox {
    int x;
    int y;
    int w;
    int h;
    int obj_id;
};
struct ImgAnno {
    int width;
    int height;
    std::string image_path;
    std::vector<Bndbox> bnd_boxes;
};
struct ImgInfo {
    ImgAnno anno;
    std::string image_id;
};

void parseline(const std::string& json_string, Bndbox* p_box, std::string* p_path, std::string* p_imgid, int* p_width, int* p_height) {
    std::string impath;
    std::string image_id;
    int x,y,h,w;
    int id;

    std::stringstream ss;
    ptree jtree, child_tree;
    ss<<json_string;
    boost::property_tree::json_parser::read_json(ss, jtree);
    // box info
    child_tree = jtree.get_child("box");
    x = child_tree.get<int>("x");
    y = child_tree.get<int>("y");
    w = child_tree.get<int>("w");
    h = child_tree.get<int>("h");
    child_tree = jtree.get_child("id");
    std::stringstream idstream;
    idstream<<child_tree.front().second.data();
    idstream>>id;
    // image info
    impath = jtree.get<std::string>("image_file");
    //image_id = jtree.get<std::string>("image_id");
    image_id = impath;
    child_tree = jtree.get_child("size");
    *p_width = child_tree.get<int>("width");
    *p_height = child_tree.get<int>("height");

    p_box->x = x;
    p_box->y = y;
    p_box->w = w;
    p_box->h = h;
    p_box->obj_id = id;
    *p_path = impath;
    *p_imgid = image_id;
}

void GatherData(const std::string &filename, std::map<std::string, ImgAnno>* p_data) {
    std::string buffer;
    std::ifstream infile(filename.c_str());
	LOG(INFO) << "Reading data from file " << filename;
    while(std::getline(infile, buffer)) {
        Bndbox box;
        std::string impath;
        int image_width, image_height;
        std::string image_id;
        parseline(buffer, &box, &impath, &image_id, &image_width, &image_height);
        if (p_data->count(image_id))
            (*p_data)[image_id].bnd_boxes.push_back(box);
        else {
            (*p_data)[image_id] = ImgAnno();
            (*p_data)[image_id].image_path = impath;
            (*p_data)[image_id].width = image_width;
            (*p_data)[image_id].height = image_height;
            (*p_data)[image_id].bnd_boxes.push_back(box);
        }
    }
	LOG(INFO) << "Image number is " << p_data->size();
}

void OrderData(const std::map<std::string, ImgAnno>& data, std::vector<shared_ptr<ImgInfo> >* p_annotations) {
    for (std::map<std::string, ImgAnno>::const_iterator itor = data.begin(); itor != data.end(); ++itor) {
        shared_ptr<ImgInfo> p_img_info(new ImgInfo);
        p_img_info->anno = itor->second;
        p_img_info->image_id = itor->first;
        p_annotations->push_back(p_img_info);
    }
    // shuffle
    if (FLAGS_shuffle) {
        LOG(INFO) << "Shuffling data";
        shuffle(p_annotations->begin(), p_annotations->end());
    }
}

void ReadData(const std::string& data_path, std::vector<shared_ptr<ImgInfo> >* annotations) {
    std::map<std::string, ImgAnno> source_data;
    GatherData(data_path, &source_data);
    OrderData(source_data, annotations);
}

int CvtDatum(const std::string& imagepath, const std::vector<Bndbox>& boxes, int width, int height, AnnotatedDatum* anno_datum) {
    // read image data, do not resize
    cv::Mat cv_img = ReadImageToCVMat(imagepath, 0, 0, 0, 0, true);
    if (!cv_img.data) {
        LOG(WARNING) << "Failed to load image data, " << imagepath;
        return 1;
    }

    bool is_encoded = true;
    if (is_encoded) {
        EncodeCVMatToDatum(cv_img, "jpg", anno_datum->mutable_datum());
    }
    else {
        CVMatToDatum(cv_img, anno_datum->mutable_datum());
    }
    anno_datum->mutable_datum()->set_label(-1);
    
    anno_datum->clear_annotation_group();
    int instance_id = 0;
    for (std::vector<Bndbox>::const_iterator itor = boxes.begin(); itor != boxes.end(); ++itor) {
        Annotation* anno = 0;
        bool found_group = false;
        int label = itor->obj_id;
        for (int g = 0; g < anno_datum->annotation_group_size(); ++g) {
            AnnotationGroup* anno_group = anno_datum->mutable_annotation_group(g);
            if (label == anno_group->group_label()) {
                if (anno_group->annotation_size() == 0) {
                    instance_id = 0;
                }
                else {
                    instance_id = anno_group->annotation(anno_group->annotation_size() - 1).instance_id() + 1;
                }
                anno = anno_group->add_annotation();
                found_group = true;
            }
        }
        if (!found_group) {
            AnnotationGroup* anno_group = anno_datum->add_annotation_group();
            anno_group->set_group_label(label);
            anno = anno_group->add_annotation();
            instance_id = 0;
        }
        anno->set_instance_id(instance_id++);
        NormalizedBBox* bbox = anno->mutable_bbox();
        bbox->set_xmin(static_cast<float>(itor->x) / width);
        bbox->set_ymin(static_cast<float>(itor->y) / height);
        bbox->set_xmax(static_cast<float>(itor->x + itor->w - 1) / width);
        bbox->set_ymax(static_cast<float>(itor->y + itor->h - 1) / height);
        bbox->set_difficult(false);
    }
    return 0;
}

int store_to_lmdb(std::string lmdb_name, \
              std::vector<shared_ptr<ImgInfo> >::const_iterator first, \
              std::vector<shared_ptr<ImgInfo> >::const_iterator last) {
    // Create new DB
    scoped_ptr<db::DB> db(db::GetDB("lmdb"));
    db->Open(lmdb_name, db::NEW);
    scoped_ptr<db::Transaction> txn(db->NewTransaction());

    // Storing to db
    AnnotatedDatum anno_datum;
    int count = 0;
    for (std::vector<shared_ptr<ImgInfo> >::const_iterator itor = first; itor != last; ++itor) {
        int status = 0;
        status = CvtDatum((*itor)->anno.image_path, (*itor)->anno.bnd_boxes, (*itor)->anno.width, (*itor)->anno.height,  &anno_datum);         
        if (status != 0) {
            continue;
            LOG(WARNING) << "Error occurred at image: " << (*itor)->anno.image_path;
        }
		anno_datum.set_type(AnnotatedDatum_AnnotationType_BBOX);

        const std::string* const p_imgpath = &((*itor)->anno.image_path);
        string key_str = caffe::format_int(count, 8) + "_" + p_imgpath->substr(p_imgpath->rfind('/')+1);
        string out;
        CHECK(anno_datum.SerializeToString(&out));
        txn->Put(key_str, out);
        
        if (++count % 1000 == 0) {
            txn->Commit();
            txn.reset(db->NewTransaction());
        }
    }
    if (count % 1000 !=0) {
        txn->Commit();
    }
    return count;
}

void store_to_file(const std::string& file_name, \
                   std::vector<shared_ptr<ImgInfo> >::const_iterator first, \
                   std::vector<shared_ptr<ImgInfo> >::const_iterator last) {
    std::ofstream file(file_name.c_str());
    for (std::vector<shared_ptr<ImgInfo> >::const_iterator itor = first; itor != last; ++itor) {
        ImgAnno& anno = (*itor)->anno;
        std::string& impath = anno.image_path;
        for (std::vector<Bndbox>::size_type i = 0; i < anno.bnd_boxes.size(); ++i) {
            file << impath << " " \
                 << anno.bnd_boxes[i].x << " " << anno.bnd_boxes[i].y << " " \
                 << anno.bnd_boxes[i].w << " " << anno.bnd_boxes[i].h << " " \
                 << anno.bnd_boxes[i].obj_id << "\n";
        }
    }
}

int main(int argc, char* argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::string data_dir = FLAGS_data_dir;
    if (*data_dir.rbegin() != '/')
        data_dir += "/";

    // Read data
    std::string data_path = data_dir + "/data.txt";
    CHECK(boost::filesystem::exists(data_path));
    std::vector<shared_ptr<ImgInfo> > data;
    ReadData(data_path, &data);

    std::vector<double> ratios(3);
    sscanf(FLAGS_train_val_test_ratios.c_str(), "[%lf,%lf,%lf]", &ratios[0], &ratios[1], &ratios[2]);
    std::vector<std::string> lmdb_names(3);
    lmdb_names[0] = std::string("train_db");
    lmdb_names[1] = std::string("validation_db");
    lmdb_names[2] = std::string("test_db");
    std::vector<std::string> file_names(3);
    file_names[0] = std::string("train.txt");
    file_names[1] = std::string("validation.txt");
    file_names[2] = std::string("test.txt");

    std::vector<shared_ptr<ImgInfo> >::const_iterator first = data.begin();
    double cum = 0.;
    for (int phase = 0; phase < 3; phase++) {
        cum += ratios[phase];
        std::vector<shared_ptr<ImgInfo> >::const_iterator last = data.begin() + int(data.size() * cum + 0.5);
        std::string lmdb_name = data_dir + lmdb_names[phase];
        std::string file_name = data_dir + file_names[phase];

        CHECK(!boost::filesystem::exists(lmdb_name));
        int num = store_to_lmdb(lmdb_name, first, last);
        store_to_file(file_name, first, last);
        first = last;
        LOG(INFO) << lmdb_name << " has " << num << " image data";
    }

    return 0;
}

