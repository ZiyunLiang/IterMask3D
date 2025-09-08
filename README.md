# IterMask3D
IterMask3D: Unsupervised Anomaly Detection and Segmentation with Test-Time Iterative Mask Refinement in 3D Brain MRI

This repository is the official pytorch implementation for paper: Liang et al., IterMask3D: Unsupervised Anomaly Detection and Segmentation with Test-Time Iterative Mask Refinement in 3D Brain MRI".

## Introduction:
Unsupervised anomaly detection and segmentation methods train a model to learn the training distribution as 'normal'. In the testing phase, they identify patterns that deviate from this normal distribution as 'anomalies'. 
To learn the 'normal' distribution, prevailing methods 
 corrupt the images and train a model to reconstruct them. 
During testing, the model attempts to reconstruct corrupted inputs based on the learned 'normal' distribution. 
Deviations from this distribution lead to high reconstruction errors, which indicate potential anomalies. 
However, corrupting an input image inevitably causes information loss even in normal regions, leading to suboptimal reconstruction and an increased risk of false positives.
To alleviate this, we propose $\rm{IterMask3D}$, an iterative spatial mask-refining strategy designed for 3D brain MRI. 
We iteratively spatially mask areas of the image as corruption and reconstruct them, then shrink the mask based on reconstruction error. This process iteratively unmasks 'normal' areas to the model, whose information further guides reconstruction of `normal' patterns under the mask to be reconstructed accurately, reducing false positives. 
In addition, to achieve better reconstruction performance, we also propose using high-frequency image content as additional structural information to guide the reconstruction of the masked area.
Extensive experiments on the detection of both synthetic and real-world imaging artifacts, as well as segmentation of various pathological lesions across multiple MRI sequences, consistently demonstrate the effectiveness of our proposed method. 

[//]: # 
![Image text](https://github.com/ZiyunLiang/IterMask3D/blob/master/img/main_figure.png)

## Usage:

### 1. preparation
**1.1 Environment**
We recommand you using conda for installing the depandencies.
The following command will help you create a new conda environment will all the required libraries installed: 
```
conda env create -f itermask3d.yml
conda activate itermask3d
```
For manualy installing packages:
- `python`                 3.9
- `torch`                  2.6.0
- `numpy`                  1.26.3
- `scipy`                  1.13.1
- `monai`                  1.0.1
- `tensorboard`            2.19.0
- `nibabel`                5.3.2
- `matplotlib`             3.9.4
- `torchio`                0.20.21

The project can be cloned using:
```
https://github.com/ZiyunLiang/IterMask3D.git
```

### 2. Dataset Preprocessing
After downloading the dataset, we first perform resampling to ensure a
standardized voxel size spacing of [1, 1, 1].
We next crop the image to a uniform size of [192, 192, 192].
Skull-stripping is then applied using Robex. Finally, the normalization is shown in normalization.py.

### Training
The testing script is in `train.py`. All arguments are in `argumentlib.py`.
The input to the model is of size [192, 192, 192].
  - `--gpu_id` Allows you to choose the GPU that you want to use for this experiment. Default: '0'
  - `--num_workers` Allows you to choose the number of workers. Default: 8
  - `--test_batch_size` Testing batch size. Default: 1
  - `--train_modality` The modality used for training. Default: 'flair'
  - `--train_data_path` The directory of the already preprocessed data and the model will load training data from this directory. Default: './datasets/data'
  - `--train_file_name_txt` If you don't want to use all the files from the testing dataset directory, this is a list of file names to use. If none, then all the files in the directory will be used. An example of the txt file is shown in the brats_split_testing_example.txt. Default: 'None' 
  - `--save_name` Model data for saving the trained model
  - `--epochs` Maximum training epochs
  - `--save_epoch` save the model every x number of epoch
  - `--lr` learning rate
  - `--train_batch_size` Training batch size. Default: 2
  - `--test_batch_size` Testing batch size. Default: 1
  - `--drop_learning_rate_epoch` At which epoch the learning rate starts to drop
  - `--drop_learning_rate_value` Learning rate decays to this value

### Testing
The testing script is in `test.py`. All arguments are in `argumentlib.py`.
The arguments used for this script are:
  - `--gpu_id` Allows you to choose the GPU that you want to use for this experiment. Default: '0'
  - `--num_workers` Allows you to choose the number of workers. Default: 8
  - `--test_batch_size` Testing batch size. Default: 2'
  - `--test_modality` The modality used for testing. Default: 'flair'
  - `--text_file_name_txt` If you don't want to use all the files from the testing dataset directory, this is a list of files to use. An example of the txt file is shown in the brats_split_testing_example.txt. Default: 'None' 
  - `--test_data_path` The directory of the already preprocessed data and the model will load data from this directory. Default: './datasets/data'
  - `--load_model_path` The path of the trained model. Default: 'None'
  - `--gamma` The first derivative gamma that is used to decide the threshold for shrinking. Default 0.05
  - `testing_task` Choose from detection or segmentation.
  - `fit_funtion_y_limit` The fit function only fit points below certain limit.
  - `fit_function_plot` If or not you want to plot the fit function to see how it performs, choose from True and False, default is True. If the fit function does not perform well, please try to adjust `fit_funtion_y_limit`. (Please focus on plotting the lower threshold values to evaluate how well they are fitted. If the curve is plotted using the full range up to the maximum threshold, the lower (and more important) values may become visually compressed, making it difficult to assess their fit accurately. Currently, the model plots threshold values (y-axis) in the range of 0 to 2.)
  - `detection_score` For anomaly detection task, when computing the metric, how to define the anomaly score, choose from final_mask_size or mean_error
  - `test_data_path2` If you need to load a second dataset for anomaly detection, the path for the second dataset
  - `shrinking_start_mask` whether the shrinking process should be initialized from the entire brain mask or from the optimal thresholded mask, choose from brain_mask or best_threshold_mask
  - `test_file_name_txt2` If you don't want to use all the files from the second testing dataset directory, this is a list of files to use. An example of the txt file is shown in the brats_split_testing_example.txt. Default: 'None' 
  - `synthetic_anomaly` If you want to generate synthetic anomaly for anomaly detection, choose from True and False
  - `synthetic_anomaly_type` If you want to generate synthetic anomaly, which synthetic anomaly you want to use, choose from 'Gaussian_noise', 'periodic_line', 'ring', 'top_chunk', 'middle_chunk'
  
Several hyperparameters that you can change with your dataset or need. 
1. `gamma` The first derivative value gamma that is used to decide the threshold for shrinking.
2. `fit_function_y_limit` To fit the fit_function with the curve, we exclude the extreme points below a certain value (the argument level). To check how is the fit function, set `fit_function_plot` to True
3. `shrinking_start_mask` If, after identifying the optimal threshold, you wish to continue shrinking the mask starting from the corresponding threshold-derived mask, set this to best_threshold_mask, default is start shrinking from the whole brain (brain_mask). Recommandaton, for 3D data, choose brain_mask, for 2D data, choose best_threshold_mask



### Citation
If you have any questions, please contact Ziyun Liang (ziyun.liang@eng.ox.ac.uk) and I am happy to discuss more with you. If you find this work helpful for your project, please give it a star and a citation. We greatly appreciate your acknowledgment.
```
@article{liang2025itermask3d,
  title={IterMask3D: Unsupervised Anomaly Detection and Segmentation with Test-Time Iterative Mask Refinement in 3D Brain MR},
  author={Liang, Ziyun and Guo, Xiaoqing and Xu, Wentian and Ibrahim, Yasin and Voets, Natalie and Pretorius, Pieter M and Noble, J Alison and Kamnitsas, Konstantinos},
  journal={arXiv preprint arXiv:2504.04911},
  year={2025}
}
```

## License
This project is licensed under the terms of the MIT license.
MIT License

Copyright (c) 2025 Ziyun Liang

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

