Open-Source Toolbox for Object Detection Metrics
================================================

Our  previously  available  [tool  for  object  detection assessment](https:// github.com/ rafaelpadilla/ Object-Detection-Metrics) has received many positive feedbacks, which motivated us to upgrade it with other metrics and support more bounding box formats. As some external tools, competitions and works are already using the older version, we decided not to modify it but release a newer and more complete project. 

The motivation of this project is the lack of consensus used by different works and implementations concerning the evaluation metrics of the object detection problem. Although on-line competitions use their own metrics to evaluate the task of object detection, just some of them offer reference code snippets to calculate the assertiveness of the detected objects.
Researchers who want to evaluate their work using different datasets than those offered by the competitions, need to implement their own version of the metrics or spend a considerable amount of time converting bounding box to formats that are supported by evaluation tools. Sometimes a wrong or different implementation can create different and biased results. Even though many tools have been developed to convert the annotated boxes from one type to another, the quality assessment of the final detections still lacks a tool compatible with different bounding box formats and multiple performance metrics. 

Ideally, in order to have trustworthy benchmarking among different approaches, it is necessary to have an implementation that can be used by everyone regardless the dataset used. This work attempts to cover this gap, providing an open-source tool flexible to support many bounding box formats and evaluate detections with different metrics (AP@[.5:.05:.95], AP@50, mAP, AR<sub>1</sub>, AR<sub>10</sub>, AR<sub>100</sub>, etc). We also provide a detailed explanation pointing out their divergences, showing how different implementations may result into different results.


## Table of contents

- [Supported bounding box formats](#different-bb-formats)
- [Important definitions](#important-definitions)
- [Metrics](#metrics)
  - [Precision x Recall curve](#precision-x-recall-curve)
  - [Average Precision](#average-precision)
    - [N-point interpolation](n-point-interpolation)
    - [Interpolating all  points](#interpolating-all-points)
  - [Mean Average Precision (mAP)](#mean-average-precision-map)
  - [Average recall](#average-recall)
  - [Mean Average Recall (mAR)](#mean-average-recall-mar)
- [A practical example](#a-practical-example)
- [Computing different metrics](#computing-different-metrics)
- [Tube Average Precision (TAP)](#tube-average-precision-tap)
- [How to use this project](#how-to-use-this-project)
- [Contributing](#how-to-contribute)
- [References](#references)



## Supported bounding box formats

This implementation does not require modifications of the detection models to match complicated input formats, avoiding conversions to XML, JSON, CSV, or other file types. It supports more than 8 different kinds of annotation formats, including the ones presented in the Table below. 

|             Annotation tool            |       Annotation types      |                                               Output formats                                               |
|:--------------------------------------:|:---------------------------:|:----------------------------------------------------------------------------------------------------------:|
|                [Label me](https://github.com/wkentaro/labelme)                | Bounding boxes and polygons |                           Labelme format, but provides conversion to COCO and PASCAL VOC                          |
|                [LabelIMG](https://github.com/tzutalin/labelImg)                |        Bounding boxes       |                                             PASCAL VOC and YOLO                                            |
|             [Microsoft VoTT](https://github.com/Microsoft/VoTT)             | Bounding boxes and polygons | PASCAL VOC, TFRecords, specific CSV, Azure Custom Vision Service, Microsoft Cognitive Toolkit (CNTK), VoTT |
| [Computer Vision Annotation Tool (CVAT)](https://github.com/openvinotoolkit/cvat) | Bounding boxes and polygons |                            COCO, CVAT, Labelme, PASCAL VOC, TFRecord, YOLO, etc                            |
|     [VGG Image Annotation Tool (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/)    | Bounding boxes and polygons |                                       COCO and specific CSV and JSON                                       |



To ensure the accuracy of the results, the implementation strictly followed the metric definitions and the output results were carefully compared against the official implementations


## Important definitions
## Metrics
### Precision x Recall curve
### Average Precision
#### N-point interpolation
#### Interpolating all points
### Mean Average Precision (mAP)
### Average recall
### Mean Average Recall (mAR)
## A practical example
## Computing different metrics
## **Tube Average Precision (TAP)**
## How to use this project

### Requirements
We highly suggest you to create an [anaconda](https://docs.anaconda.com/anaconda/install/) environment using the `environment.yml` file available in this repository. To create the environment and install all necessary packages, run the following command:

`conda env create -n <env_name> --file environment.yml`

Now activate the evironment

`conda activate <env_name>`

And you are ready to run!

### Running

To help users to apply different metrics using multiple bounding box formats, a GUI was created to facilitate the evaluation process. By running the command `python run.py`, the following screen will show:

<!--- interpolated precision AUC --->
<p align="center">
<img src="https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/data/images/printshot_main_screen.png" align="center"/>
</p>

Each number in red represents a funcionality described below:

1) **Annotations**: Select the folder containing the ground-truth annotation file(s). 
2) **Images**: Select the folder containing the images. This is only necessary  if your ground-truth file contains formats in relative coordinates and to visualize the images (see item 5).  
3) **Classes**: YOLO (.txt) training format represents the classes with IDs (sequential integers). For this annotation type, you need to inform a txt file listing one class per line. The first line refers to the class with id 0, the second line is the class with id 1, and so on. See [here](https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/toyexample/voc.names) an example of file used to represent classes of the VOC PASCAL dataset. 
4) **Coordinate formats**: Choose the format of the annotations file(s). 
5) **Ground-truth statistics**: This is an optional feature that provides the amount of bounding boxes of each ground-truth class and to visualize the images with bounding boxes. To access this option, you must have selected the images (see item 2).   
6) **Annotations**: Select the folder containing the annotation file(s) with your detections.  
7) **Classes**: If your coordinats formats represent the classes with IDs (sequential integers), you need to inform a text file listing one class per line. The first line refers to the class with id 0, the second line is the class with id 1, and so on. See [here](https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/toyexample/voc.names) an example of file used to represent classes of the VOC PASCAL dataset. 
8) **Coordinate formats**: Choose the format of the files used to represent the the detections.  
9) **Detections statistics**: This is an optional feature that provides the amount of detections per class. You can also visualize the quality of the detections by plotting the detected and ground-truth boxes on the images.  
10) **Metrics**: Select at least one metric to evaluate your detections. For the PASCAL VOC AP and mAP, you can choose different IOUs. Note that the default IOU threshold used in the PASCAL VOC AP metric is 0.5. 
11) **Output**: Choose a folder where PASCAL VOC AP plots will be saved.  
12) **RUN**: Run the metrics. Depending on the amount of your dataset and the format of your detections, it may take a while. Detections in relative coordinates usually take a little longer to read than other formats. 

Visualize the statistics of your dataset (Options #5 and #9: Ground-truth and detection statistics) to make sure you have chosen the right formats. If somehow the formats are incorrect the boxes are going to appear incorreclty on the images. 

<!--- interpolated precision AUC --->
<p align="center">
<img src="https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/data/images/printshot_details_groundtruth.png" align="center"/>
</p>

You can also save the images and plot a bar plot with the distribution of the boxes per class. 

## Contributing

We appreciate all contributions. If you are planning to contribute with this repository, please do so without any further discussion.

If you plan to add new features, support other bounding box formats, create tutorials, please first open an issue and discuss the feature with us. If you send a PR without previous discussion, it might be rejected.

It is also important that for each new feature, supporting other bounding box formats, and metrics, a pytest must be created.

## References

