

<p align="left">
  
<a>[![Build Status](https://travis-ci.com/rafaelpadilla/review_object_detection_metrics.svg?branch=main)](https://travis-ci.com/rafaelpadilla/review_object_detection_metrics)</a>
<a href="https://github.com/rafaelpadilla/review_object_detection_metrics/raw/main/published_paper.pdf">
    <img src="https://img.shields.io/badge/paper-published-blue"/></a>
<a><img src="https://img.shields.io/badge/version-0.1-orange"/></a>
<a href="https://doi.org/10.3390/electronics10030279">
    <img src="https://img.shields.io/badge/DOI-10.3390%2Felectronics10030279-gray"/></a>
</p>

## Citation

This work was published in the [Journal Electronics - Special Issue Deep Learning Based Object Detection](https://www.mdpi.com/2079-9292/10/3/279). 

If you use this code for your research, please consider citing:

```
@Article{electronics10030279,
AUTHOR = {Padilla, Rafael and Passos, Wesley L. and Dias, Thadeu L. B. and Netto, Sergio L. and da Silva, Eduardo A. B.},
TITLE = {A Comparative Analysis of Object Detection Metrics with a Companion Open-Source Toolkit},
JOURNAL = {Electronics},
VOLUME = {10},
YEAR = {2021},
NUMBER = {3},
ARTICLE-NUMBER = {279},
URL = {https://www.mdpi.com/2079-9292/10/3/279},
ISSN = {2079-9292},
DOI = {10.3390/electronics10030279}
}
```
Download the paper [here](https://www.mdpi.com/2079-9292/10/3/279/pdf) or [here](https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/published_paper.pdf).


Open-Source Visual Interface for Object Detection Metrics
================================================

Our  [previously available  tool](https://github.com/rafaelpadilla/Object-Detection-Metrics) for  object  detection assessment has received many positive feedbacks, which motivated us to upgrade it with other metrics and support more bounding box formats. As some external tools, competitions and works are already using the older version, we decided not to modify it but release a newer and more complete project.

The motivation of this project is the lack of consensus used by different works and implementations concerning the evaluation metrics of the object detection problem. Although on-line competitions use their own metrics to evaluate the task of object detection, just some of them offer reference code snippets to calculate the assertiveness of the detected objects.
Researchers, who want to evaluate their work using different datasets than those offered by the competitions, need to implement their own version of the metrics or spend a considerable amount of time converting their bounding boxes to formats that are supported by evaluation tools. Sometimes a wrong or different implementation can create different and biased results. Even though many tools have been developed to convert the annotated boxes from one type to another, the quality assessment of the final detections still lacks a tool compatible with different bounding box formats and multiple performance metrics.

Ideally, in order to have trustworthy benchmarking among different approaches, it is necessary to have an implementation that can be used by everyone regardless the dataset used. This work attempts to cover this gap, providing an open-source tool flexible to support many bounding box formats and evaluate detections with different metrics (AP@[.5:.05:.95], AP@50, mAP, AR<sub>1</sub>, AR<sub>10</sub>, AR<sub>100</sub>, etc). We also provide a detailed explanation pointing out their divergences, showing how different implementations may result into different results.


## Table of contents

- [Open-Source Toolbox for Object Detection Metrics](#open-source-toolbox-for-object-detection-metrics)
  - [Table of contents](#table-of-contents)
  - [Supported bounding box formats](#supported-bounding-box-formats)
  - [A practical example](#a-practical-example)
  - [Metrics](#metrics)
    - [AP with IOU Threshold *t=0.5*](#ap-with-iou-threshold-t05)
    - [mAP with IOU Threshold *t=0.5*](#map-with-iou-threshold-t05)
    - [AP@.5 and AP@.75](#ap5-and-ap75)
    - [AP@[.5:.05:.95]](#ap50595)
    - [AP<sub>S</sub>, AP<sub>M</sub> and AP<sub>L</sub>](#AP-s-AP-M-AP-L)  
  - [**Spatio-Temporal Tube Average Precision (STT-AP)**](#spatio-temporal-tube-average-precision-stt-ap)
  - [How to use this project](#how-to-use-this-project)
    - [Requirements](#requirements)
    - [Running](#running)
      - [Images](#images)
      - [Spatio-Temporal Tube](#spatio-temporal-tube)
  - [Contributing](#contributing)


## Supported bounding box formats

This implementation does not require modifications of the detection models to match complicated input formats, avoiding conversions to XML, JSON, CSV, or other file types. It supports more than 8 different kinds of annotation formats, including the most popular ones, as presented in the Table below.

|                                  Annotation tool                                  |      Annotation types       |                                               Output formats                                               |
| :-------------------------------------------------------------------------------: | :-------------------------: | :--------------------------------------------------------------------------------------------------------: |
|                  [Label me](https://github.com/wkentaro/labelme)                  | Bounding boxes and polygons |                       Labelme format, but provides conversion to COCO and PASCAL VOC                       |
|                 [LabelIMG](https://github.com/tzutalin/labelImg)                  |       Bounding boxes        |                                            PASCAL VOC and YOLO                                             |
|                [Microsoft VoTT](https://github.com/Microsoft/VoTT)                | Bounding boxes and polygons | PASCAL VOC, TFRecords, specific CSV, Azure Custom Vision Service, Microsoft Cognitive Toolkit (CNTK), VoTT |
| [Computer Vision Annotation Tool (CVAT)](https://github.com/openvinotoolkit/cvat) | Bounding boxes and polygons |                            COCO, CVAT, Labelme, PASCAL VOC, TFRecord, YOLO, etc                            |
| [VGG Image Annotation Tool (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/) | Bounding boxes and polygons |                                       COCO and specific CSV and JSON                                       |

## A practical example

Considering the set of 12 images in the figure below:

<!--- Toy example figure --->
<p align="center">
<img src="https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/data/images/toy_example_mosaic.png" align="center" width="1000"/>
</p>

Each image, except (a), (g), and (j), has at least one target object of the class *cat*, whose locations are limited by the green rectangles.
There is a total of 12 target objects limited by the green boxes. Images (b), (e), and (f) have two ground-truth samples of the target class.
An object detector predicted 12 objects represented by the red rectangles (labeled with letters *A* to *L*) and their associated confidence levels are represented in percentages. Images (a), (g), and (j) are expected to have no detection. Conversely, images (b), (e), and (f) have two ground-truth bounding boxes.

To evaluate the precision and recall of the 12 detections it is necessary to establish an IOU threshold *t*, which will classify each detection as TP or FP.
In this example, let us first consider as TP the detections with *IOU > 50%*, that is *t=0.5*.

<!--- Toy example table t=0.5 --->
<p align="center">
<img src="https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/data/images/table_1_toyexample.png" align="center" width="700"/>
</p>

As stated before, AP is a metric to evaluate precision and recall in different confidence values. Thus, it is necessary to count the amount of TP and FP classifications given different confidence levels.

By choosing a more restrictive IOU threshold, different precision x recall values can be obtained. The following table computes the precision and recall values with a more strict IOU threshold of *t = 0.75*. By that, it is noticeable the occurrence of more FP detections, reducing the recall.

<!--- Toy example table t=0.75 --->
<p align="center">
<img src="https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/data/images/table_2_toyexample.png" align="center" width="720"/>
</p>

Graphical representations of the precision x values presented in both cases *t= 0.5* and *t=0.75* are shown below:

<!--- Curves t=0.5 and t=0.75 --->
<p align="center">
<img src="https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/data/images/precision_recall_curve_toyexample.png" align="center"/>
</p>


By comparing both curves, one may note that for this example:

1) With a less restrictive IOU threshold (*t=0.5*), higher recall values can be obtained with the highest precision. In other words, the detector can retrieve about *66.5%* of the total ground truths without any miss detection.
2) Using *t=0.75*, the detector is more sensitive with different confidence values. This is explained by the amount of ups and downs of the curve.
3) Regardless the IOU threshold applied, this detector can never retrieve *100%* of the ground truths (recall = 1). This is due to the fact that the algorithm did not predict any bounding box for one of the ground truths in image (e).

Different methods can be applied to measure the AUC of the precision x recall curve. Considering the *N-point interpolation* to calculate the AP with *N=11*, the interpolation measures the recall in the points L=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], and considering the *All-point interpolation* approach, all points are considered. Both approaches result in different plots as shown below:


<!--- Interpolating curves --->
<p align="center">
<img src="https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/data/images/interpolations_toyexample.png" align="center"/>
</p>

When an IOU threshold *t=0.5* was applied (plots of the first row in image above), the 11-point interpolation method obtained *AP=88.64%* while the all-point interpolation method improved the results a little, reaching *AP=89.58%*. Similarly, for an IOU threshold of *t=0.75%* (plots of the second row in image above), the 11-point interpolation method obtained *AP=49.24%* and the all-point interpolation *AP=50.97%*.

In both cases, the all-point interpolation approach considers larger areas above the curve into the summation and consequently obtains higher results.
When a lower IOU threshold was considered, the AP was reduced drastically in both interpolation approaches. This is caused by the flexibility the threshold brings in considering TP detections.

## Metrics

As each dataset adopts a specific annotation format, works tend to use the evaluation tools provided by the datasets considered to test the performance of their methods, what makes their results dependent to the implemented metric type. PASCAL VOC dataset uses the PASCAL VOC annotation format, which provides a MATLAB evaluation code of the metrics AP and mAP (IOU=.50) hampering other types of metrics to be reported with this dataset. The following table shows that among the listed methods, results are reported using a total of 14 different metrics. Due to the fact that the evaluation metrics are directly associated with the annotation format of the datasets, almost all works report their results using only the metrics implemented by the benchmarking datasets, making such cross-datasets comparative results quite rare in the object detection literature.

|    Method    |             Benchmark dataset              |                                     Metrics                                     |
| :----------: | :----------------------------------------: | :-----------------------------------------------------------------------------: |
|  CornerNet   |                    COCO                    |                 AP@[.5:.05:.95]; AP@.50; AP@.75; APS; APM; APL                  |
| EfficientDet |                    COCO                    |                         AP@[.5:.05:.95]; AP@.50; AP@.75                         |
|  Fast R-CNN  |        PASCAL VOC 2007, 2010, 2012         |                                AP; mAP (IOU=.50)                                |
| Faster R-CNN |           PASCAL VOC 2007, 2012            |                                AP; mAP (IOU=.50)                                |
| Faster R-CNN |                    COCO                    |                             AP@[.5:.05:.95]; AP@.50                             |
|    R-CNN     |        PASCAL VOC 2007, 2010, 2012         |                                AP; mAP (IOU=.50)                                |
|   RFB Net    |                  VOC 2007                  |                                  mAP (IOU=.50)                                  |
|   RFB Net    |                    COCO                    |                 AP@[.5:.05:.95]; AP@.50; AP@.75; APS; APM; APL                  |
|  RefineDet   |               VOC 2007, 2012               |                                  mAP (IOU=.50)                                  |
|  RefineDet   |                    COCO                    |                 AP@[.5:.05:.95]; AP@.50; AP@.75; APS; APM; APL                  |
|  RetinaNet   |                    COCO                    |                 AP@[.5:.05:.95]; AP@.50; AP@.75; APS; APM; APL                  |
|    R-FCN     |               VOC 2007, 2012               |                                  mAP (IOU=.50)                                  |
|    R-FCN     |                    COCO                    |                      AP@[.5:.05:.95];AP@.50; APS; APM; APL                      |
|     SSD      |               VOC 2007, 2012               |                                  mAP (IOU=.50)                                  |
|     SSD      |                    COCO                    | AP@[.5:.05:.95]; AP@.50; AP@.75; APS; APM; APL; AR1; AR10; AR100; ARS; ARM; ARL |
|     SSD      |                  ImageNet                  |                                  mAP (IOU=.50)                                  |
|   Yolo v1    | PASCAL VOC 2007, 2012; Picasso; People-Art |                                AP; mAP (IOU=.50)                                |
|   Yolo v2    |           PASCAL VOC 2007, 2012            |                                AP; mAP (IOU=.50)                                |
|   Yolo v2    |                    COCO                    | AP@[.5:.05:.95]; AP@.50; AP@.75; APS; APM; APL; AR1; AR10; AR100; ARS; ARM; ARL |
|   Yolo v3    |                    COCO                    | AP@[.5:.05:.95]; AP@.50; AP@.75; APS; APM; APL; AR1; AR10; AR100; ARS; ARM; ARL |
|   Yolo v4    |                    COCO                    |                 AP@[.5:.05:.95]; AP@.50; AP@.75; APS; APM; APL                  |
|   Yolo v5    |                    COCO                    |                             AP@[.5:.05:.95]; AP@.50                             |


As previously presented, there are different ways to evaluate the area under the precision x recall and recall x IOU curves. Nonetheless, besides the combinations of different IOU thresholds and interpolation points, other considerations are also applied resulting in different metric values. Some methods limit the evaluation by object scales and detections per image. Such variations are computed and named differently as shown below:

### AP with IOU Threshold *t=0.5*

This AP metric is widely used to evaluate detections in the PASCAL VOC dataset. It measures the AP of each class individually by computing the area under the precision x recall curve interpolating all points. In order to classify detections as TP or FP the IOU threshold is set to *t=0.5*.

### mAP with IOU Threshold *t=0.5*

This metric is also used by PASCAL VOC dataset and is calculated as the AP with IOU *t=0.5*, but the result obtained by each class is averaged.

### AP@.5 and AP@.75

These two metrics evaluate the precision x curve differently than the PASCAL VOC metrics. In this method, the interpolation is performed in *N=101* recall points. Then, the computed results for each class are summed up and divided by the number of classes.

The only difference between AP@.5 and AP@.75 is the applied IOU thresholds. AP@.5 uses *t=0.5* whereas AP@.75 applies *t=0.75*. These metrics are commonly used to report detectAP<sub>S</sub>, AP<sub>M</sub> and AP<sub>L</sub>ions performed in the COCO dataset.

### AP@[.5:.05:.95]

This metric expands the AP@.5 and AP@.75 metrics by computing the AP@ with ten different IOU thresholds (*t=[0.5, 0.55, ..., 0.95]*) and taking the average among all computed results.

### AP<sub>S</sub>, AP<sub>M</sub> and AP<sub>L</sub> <a name="AP-s-AP-M-AP-L">

These three metrics, also referred to as AP Across Scales, apply the AP@[.5,.05:.95] taking into consideration the size of the ground-truth object. AP<sub>S</sub> only evaluates the ground-truth objects of small sizes (area < 32^2 pixels); AP<sub>M</sub> considers only ground-truth objects of medium sizes (32^2 < area < 96^2 pixels); AP<sub>L</sub> considers large ground-truth objects (area > 96^2) only.

When evaluating objects of a given size, objects of the other sizes (both ground-truth and predicted) are not considered in the evaluation. This metric is also part of the COCO evaluation dataset.


## **Spatio-Temporal Tube Average Precision (STT-AP)**
When dealing with videos, one may be interested in evaluating the model performance at  video level, i.e., whether the object was detected in the video as a whole. This metric is an extension of the AP metric that integrates spatial and temporal localizations; it is concise, yet expressive.
A spatio-temporal tube *To* of an object *o* is the spatio-temporal region defined as the concatenation of the bounding boxes of an object from each frame of a video, that is *T_o = [B<sub>o,q</sub> B<sub>o,q+1</sub> ... B<sub>o,q+Q-1</sub>*, where *B<sub>o,k</sub>* is the bounding box of the object *o* in frame *k* of the video that is constituted of *Q* frames indexed by *k= q, q+1,..., q+Q-1*.
Considering a ground-truth spatio-temporal tube *T<sub>gt</sub>* and a predicted spatio-temporal tube *T<sub>p</sub>*, the spatio-temporal tube IOU (STT-IOU) measures the ratio of the overlapping to the union of the "discrete volume" between *T<sub>gt</sub>* and *T<sub>p</sub>*,
as illustrated bellow:
<!--- STT_IOU figure --->
<p align="center">
<img src="https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/data/images/STT_IOU.png" align="center" width="380" />
</p>

Based on these definitions, the proposed STT-AP metric follows the AP.

## How to use this project

### Requirements

We highly suggest you to create an [anaconda](https://docs.anaconda.com/anaconda/install/) environment using the `environment.yml` file available in this repository. To create the environment and install all necessary packages, run the following command:

`conda env create -n <env_name> --file environment.yml`  

Now activate the evironment: `conda activate <env_name>`  

Install the tool: `python setup.py install`  

Run the UI: `python run.py`  

### Running

#### Images
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

#### Spatio-Temporal Tube

##### Ground-truth Format
For annotation with STT, use a .json file following format:

```
{
"videos": [
  {
    "id": int,
    "file_name": str,
    "width": int,
    "height": int
  }
]

"annotations": [
  {
    "id": int,
    "video_id": int,
    "category_id": int,
    "track":[
      {
        "frame": int,
        "bbox": [x ,y , width, height],
        "confidence": float
      }
    ]
  }]

"categories": [
  {
    "id": int,
    "name": str
  }
]
}
```

##### Predictions Format
For detection with STT, use a .json file following format:

```
[
  {
    "id": int,
    "video_id": int,
    "category_id": int,
    "track":[
      {
        "frame": int,
        "bbox": [x ,y , width, height],
        "confidence": float
      }
    ]
  }
]
```
See [example annotation](https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/tests/tube/example_anno.json)  and [example predictions](https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/tests/tube/example_preds.json) for examples of annotation and prediction .json files.

##### Running  

```python
from src.evaluators.tube_evaluator import TubeEvaluator

tube_evaluator = TubeEvaluator(annot_filepath, preds_filepath)
res, mAP = tube_evaluator.evaluate(thr=0.5)
```

## Contributing

We appreciate all contributions. If you are planning to contribute with this repository, please do so without any further discussion.

If you plan to add new features, support other bounding box formats, create tutorials, please first open an issue and discuss the feature with us. If you send a PR without previous discussion, it might be rejected.

It is also important that for each new feature, supporting other bounding box formats, and metrics, a pytest must be created.

