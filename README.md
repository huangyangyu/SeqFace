# SeqFace : Making full use of sequence information for face recognition

  Paper link: https://arxiv.org/abs/1803.06524

    Paper by Wei Hu, Yangyu Huang, Fan Zhang, Ruirui Li, Wei Li, Guodong Yuan


### Recent Update

  **`next time`**: Coming soon.

  **`2018.03.20`**: 1.Publish our paper; 2.Release test dataset and test code.

  **`2018.03.15`**: 1.Create the repository; 2.Release our model. 


### Contents
0. [Requirements](#requirements)
0. [Dataset](#dataset)
0. [Model-and-Result](#model-and-result)
0. [Demo](#demo)
0. [Contact](#contact)
0. [Citation](#citation)
0. [License](#license)


### Requirements

  1. **`Caffe`** (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  2. **`MTCNN`** (see: [MTCNN - face detection & alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment))


### Dataset

  All faces in our dataset are detected by MTCNN and aligned by align.py. The structure of trainning dataset and testing dataset is shown below. Please note that the testing dataset have already processed by detection and alignment, So you can get our result directly by running our evaluating script.

  Training Dataset

  [**`MS-Celeb-1M`**](https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-million-celebrities-real-world/) + **`Celeb-Seq`**

  Testing Dataset

  **`LFW`** [@BaiduDrive](https://pan.baidu.com/s/1C16_nR7C8h36kqtIhA0WLw), [@GoogleDrive](https://drive.google.com/file/d/1YXo8M51jycZeNhgVGZEeTQigopPnz2pi/view)

  **`YTF`** [@BaiduDrive](https://pan.baidu.com/s/1dBf0_e-pGLxYFN8tNf7qEA), [@GoogleDrive](https://drive.google.com/file/d/19BgCxFqMgNpczFmwnDD1eFH8hHHslmJ7/view)


### Model-and-Result

  We released our ResNet-27 model, you can download it by the link below. The model was trained in caffe, please refer to our paper for the detailed training process.

  **`Caffe: ResNet-27`** [@BaiduDrive](https://pan.baidu.com/s/1B5HCTfcYs7s-QeVeAzbNVw), [@GoogleDrive](https://drive.google.com/file/d/1Iqhn_SLpo_2QbIPxw8ht3tGo-2K5dExC/view)

  Performance:

  | Dataset  | Model        | Dataset                  | LFW    | YTF    |
  | -------- | ------------ | ------------------------ | ------ | ------ |
  | SeqFace  | 1 ResNet-27  | MS-Celeb-1M + Celeb-Seq  | 99.80  | 98.00  |
  | SeqFace  | 1 ResNet-64  | MS-Celeb-1M + Celeb-Seq  | 99.83  | 98.12  |


### Demo

  You can experience our algorithm on [demo page](http://imgserver.yunshitu.cn/verication/)


### Contact

  [Wei Hu](mailto:huwei@mail.buct.edu.cn)

  [Yangyu Huang](mailto:yangyu.huang.1990@outlook.com)


### Citation

  Waiting


### License

  SeqFace is released under the MIT License

