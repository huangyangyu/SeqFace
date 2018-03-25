# SeqFace : Making full use of sequence information for face recognition

  Paper link: https://arxiv.org/abs/1803.06524

    Paper by Wei Hu, Yangyu Huang, Fan Zhang, Ruirui Li, Wei Li, Guodong Yuan


### Recent Update

  **`2018.03.21`**: 1.Release the code of LSRLoss and DSALoss layer; 2.Add example of Seqface.

  **`2018.03.20`**: 1.Publish our paper; 2.Release test dataset and test code.

  **`2018.03.15`**: 1.Create the repository; 2.Release our model. 


### Contents
1. [Requirements](#requirements)
1. [Dataset](#dataset)
1. [Model-and-Result](#model-and-result)
1. [How-to-test](#how-to-test)
1. [Example](#example)
1. [Demo](#demo)
1. [Contact](#contact)
1. [Citation](#citation)
1. [License](#license)


### Requirements

  1. **`Caffe`** (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  2. **`MTCNN`** (see: [MTCNN - face detection & alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment))


### Dataset

  All faces in our dataset are detected by MTCNN and aligned by util.py. The structure of trainning dataset and testing dataset is shown below. Please note that the testing dataset have already be processed by detection and alignment, So you can reproduce our result directly by running our evaluating script.

  Training Dataset

  [**`MS-Celeb-1M`**](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/) + **`Celeb-Seq`**

  Testing Dataset

  **`LFW`** [@BaiduDrive](https://pan.baidu.com/s/1C16_nR7C8h36kqtIhA0WLw), [@GoogleDrive](https://drive.google.com/file/d/1YXo8M51jycZeNhgVGZEeTQigopPnz2pi/view)

  **`YTF`** [@BaiduDrive](https://pan.baidu.com/s/1dBf0_e-pGLxYFN8tNf7qEA), [@GoogleDrive](https://drive.google.com/file/d/19BgCxFqMgNpczFmwnDD1eFH8hHHslmJ7/view)

  Testing Features

  You can also use the precomputed feature instead of the testing dataset to evaluate our method.

  **`LFW`**: [@BaiduDrive](https://pan.baidu.com/s/1IMNqie0KRdfFF4InOF2qVw), [@GoogleDrive](https://drive.google.com/file/d/1ChQN4sPI7eqSvdaztLn_Px4i656IPwtz/view)

  **`YTF`**: [@BaiduDrive](https://pan.baidu.com/s/1JSoNKbgPzEh984aRlcLpmQ), [@GoogleDrive](https://drive.google.com/file/d/1CI10KoHm2Te62678kFvnV1vRVboQVqjf/view)


### Model-and-Result

  We released our ResNet-27 model, which can be downloaded by the link below. The model was trained in caffe, please refer to our paper for the detailed training process.

  **`Caffe: ResNet-27`** [@BaiduDrive](https://pan.baidu.com/s/1B5HCTfcYs7s-QeVeAzbNVw), [@GoogleDrive](https://drive.google.com/file/d/1Iqhn_SLpo_2QbIPxw8ht3tGo-2K5dExC/view) (non-commercial use only)

  Performance:

  | Dataset  | Model        | Dataset                  | LFW    | YTF    |
  | -------- | ------------ | ------------------------ | ------ | ------ |
  | SeqFace  | 1 ResNet-27  | MS-Celeb-1M + Celeb-Seq  | 99.80  | 98.00  |
  | SeqFace  | 1 ResNet-64  | MS-Celeb-1M + Celeb-Seq  | 99.83  | 98.12  |


### How-to-test

  Refer to run.sh, which contains two parameters, the first one("mode") means the running mode you use("feature" or "model"), the other one("dataset") means the dataset you choose("LFW" or "YTF").

    step 1: git clone https://github.com/ydwen/caffe-face.git

    step 2: compile caffe

    step 3: download model and testing dataset, then unzip them

    step 4: run evaluate.py in LFW or YTF directory

  You can try the command below to verify seqface of LFW dataset in feature mode.

    ./run.sh feature LFW


### Example

  ![image](https://github.com/huangyangyu/SeqFace/blob/master/example/framwork.png)

  The usage of LSR loss layer and DSA loss layer in train_val.prototxt:

    layer {
        name: "lsro_loss"
        type: "LSROLoss"
        bottom: "fc6"
        bottom: "label000"
        top: "lsro_loss"
        loss_weight: @loss_weight
    }

    layer {
        name: "dsa_loss"
        type: "DSALoss"
        bottom: "fc5"
        bottom: "label000"
        top: "dsa_loss"
        param {
            lr_mult: 1
            decay_mult: 0
        }
        dsa_loss_param {
            faceid_num: @faceid_num
            seqid_num: @seqid_num
            center_filler {
                type: "xavier"
            }
            lambda: @lambda
            dropout_ratio: @dropout_ratio
            gamma: @gamma
            alpha: @alpha
            beta: @beta
            use_normalize: @use_normalize
            scale: @scale
        }
        loss_weight: @loss_weight
    }


### Demo

  You can experience our algorithm on [demo page](http://imgserver.yunshitu.cn/verication/)


### Contact

  [Wei Hu](mailto:huwei@mail.buct.edu.cn)

  [Yangyu Huang](mailto:yangyu.huang.1990@outlook.com)


### Citation

  Waiting


### License

    SeqFace(not including the model) is released under the MIT License

