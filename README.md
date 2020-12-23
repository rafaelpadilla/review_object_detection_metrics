## Citation
This work was submited to the Journal [Electronics](https://www.mdpi.com/journal/electronics).

<!----
```
@INPROCEEDINGS {padillaCITE2020,
    author    = {R. {Padilla} and W. L. {Passos} and T. L. B. {Dias} and S. L. {Netto} and E. A. B. {da Silva}},
    title     = {A Survey on Object Detection Metrics with a Companion Open-Source Toolbox},
    journal   = {Electronics},
    volume    = {9},
    year      = {2020},
    pages     = {},
    publisher = {Multidisciplinary Digital Publishing Institute}}
```
Download the paper [here](TODO)
--->


Open-Source Toolbox for Object Detection Metrics
================================================

Our  [previously available  tool](https://github.com/rafaelpadilla/Object-Detection-Metrics) for  object  detection assessment has received many positive feedbacks, which motivated us to upgrade it with other metrics and support more bounding box formats. As some external tools, competitions and works are already using the older version, we decided not to modify it but release a newer and more complete project.

The motivation of this project is the lack of consensus used by different works and implementations concerning the evaluation metrics of the object detection problem. Although on-line competitions use their own metrics to evaluate the task of object detection, just some of them offer reference code snippets to calculate the assertiveness of the detected objects.
Researchers, who want to evaluate their work using different datasets than those offered by the competitions, need to implement their own version of the metrics or spend a considerable amount of time converting their bounding boxes to formats that are supported by evaluation tools. Sometimes a wrong or different implementation can create different and biased results. Even though many tools have been developed to convert the annotated boxes from one type to another, the quality assessment of the final detections still lacks a tool compatible with different bounding box formats and multiple performance metrics.

Ideally, in order to have trustworthy benchmarking among different approaches, it is necessary to have an implementation that can be used by everyone regardless the dataset used. This work attempts to cover this gap, providing an open-source tool flexible to support many bounding box formats and evaluate detections with different metrics (AP@[.5:.05:.95], AP@50, mAP, AR<sub>1</sub>, AR<sub>10</sub>, AR<sub>100</sub>, etc). We also provide a detailed explanation pointing out their divergences, showing how different implementations may result into different results.


## Table of contents

- [Open-Source Toolbox for Object Detection Metrics](#open-source-toolbox-for-object-detection-metrics)
  - [Table of contents](#table-of-contents)
  - [Supported bounding box formats](#supported-bounding-box-formats)
  - [Important definitions](#important-definitions)
    - [IOU: Intersection over union](#iou-intersection-over-union)
    - [Precision and Recall](#precision-and-recall)
    - [Average Precision (AP)](#average-precision-ap)
      - [N-point interpolation](#n-point-interpolation)
      - [All-point interpolation](#all-point-interpolation)
    - [Mean average precision (mAP)](#mean-average-precision-map)
    - [Average recall (AR)](#average-recall-ar)
    - [Mean average recall (mAR)](#mean-average-recall-mar)
  - [A practical example](#a-practical-example)
  - [Metrics](#metrics)
    - [AP with IOU Threshold *t=0.5*](#ap-with-iou-threshold-t05)
    - [mAP with IOU Threshold *t=0.5*](#map-with-iou-threshold-t05)
    - [AP@.5 and AP@.75](#ap5-and-ap75)
    - [AP@[.5:.05:.95]](#ap50595)
    - [AP\textsubscript{S}, AP\textsubscript{M}, and AP\textsubscript{L}} \label{sec:AP_sizes}](#aptextsubscripts-aptextsubscriptm-and-aptextsubscriptl-labelsecap_sizes)
    - [AR\textsubscript{1}, AR\textsubscript{10}, and AR\textsubscript{100}](#artextsubscript1-artextsubscript10-and-artextsubscript100)
    - [AR\textsubscript{S}, AR\textsubscript{M} and AR\textsubscript{L}](#artextsubscripts-artextsubscriptm-and-artextsubscriptl)
  - [**Spatio-Temporal Tube Average Precision (STT-AP)**](#spatio-temporal-tube-average-precision-stt-ap)
  - [How to use this project](#how-to-use-this-project)
    - [Requirements](#requirements)
    - [Running](#running)
      - [Images](#images)
      - [Spatio-Temporal Tube](#spatio-temporal-tube)
        - [annotation format](#annotation-format)
        - [predictions format](#predictions-format)
        - [run](#run)
  - [Contributing](#contributing)
  - [References](#references)


## Supported bounding box formats

This implementation does not require modifications of the detection models to match complicated input formats, avoiding conversions to XML, JSON, CSV, or other file types. It supports more than 8 different kinds of annotation formats, including the most popular ones, as presented in the Table below.

|                                  Annotation tool                                  |      Annotation types       |                                               Output formats                                               |
| :-------------------------------------------------------------------------------: | :-------------------------: | :--------------------------------------------------------------------------------------------------------: |
|                  [Label me](https://github.com/wkentaro/labelme)                  | Bounding boxes and polygons |                       Labelme format, but provides conversion to COCO and PASCAL VOC                       |
|                 [LabelIMG](https://github.com/tzutalin/labelImg)                  |       Bounding boxes        |                                            PASCAL VOC and YOLO                                             |
|                [Microsoft VoTT](https://github.com/Microsoft/VoTT)                | Bounding boxes and polygons | PASCAL VOC, TFRecords, specific CSV, Azure Custom Vision Service, Microsoft Cognitive Toolkit (CNTK), VoTT |
| [Computer Vision Annotation Tool (CVAT)](https://github.com/openvinotoolkit/cvat) | Bounding boxes and polygons |                            COCO, CVAT, Labelme, PASCAL VOC, TFRecord, YOLO, etc                            |
| [VGG Image Annotation Tool (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/) | Bounding boxes and polygons |                                       COCO and specific CSV and JSON                                       |

## Important definitions

In the object detection context, a detection is defined as a rectangular region of an image, represented by a **bounding box**, associated to a **class** (e.g. cat, person, truck, ball, etc) with a **confidence level**.

A detection is considered to occur whenever the probability of a given class, as output by a detector, is larger than a given threshold. Since this threshold is a probability, it also defines a confidence level for the detection.

### IOU: Intersection over union

Consider a target object to be detected represented by a ground-truth bounding box $B_{gt}$ and the detected area by an object detector represented by a predicted bounding box $B_{p}$. Without examining the confidence level, a perfect match is considered when the area and location of the predicted and ground-truth boxes are the same. These two conditions are guaranteed by the intersection over union (IOU), a measurement based on the Jaccard Index, a coefficient of similarity for two sets of data.

In the object detection scope, the IOU measures the overlapping area between the predicted bounding box $B_{p}$ and the ground-truth bounding box $B_{gt}$ divided by the area of union between them, that is:

<!--- IOU Equation --->
<p align="center">
<img src="https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/data/images/iou_eq.png" align="center"/>
</p>,
<!--- \begin{equation} \label{eqIOU} J(B_p,B_\textit{gt}) = {\rm IOU} =
\frac{\text{area}(B_p \cap B_\textit{gt}\text{)}}{\text{area}(B_p \cup B_\textit{gt})},
\end{equation} --->

and can be illustrated as:

<!--- IOU figure --->
<p align="center">
<img src="https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/data/images/iou.png" align="center" width="380" />
</p>

A perfect match occurs when IOU=1 and, if both bounding boxes do not intercept each other, IOU=0. The closer to 1 the IOU gets, the better the detection is considered. It is important to mention that as object detectors also perform the classification of each bounding box, only ground-truth and detected boxes of the same class are comparable.


### Precision and Recall

Precision is the ability of a model to identify only relevant objects. It is the percentage of correct positive predictions. Recall is the ability of a model to find all relevant cases (all ground-truth bounding boxes). It is the percentage of correct positive predictions among all given ground truths. In order to
calculate the precision and recall values, each detected bounding box must first be classified as:

• True positive (TP): A correct detection of a ground-truth bounding box;
• False positive (FP): An incorrect detection of a non-existing object or a misplaced detection of an existing object;
• False negative (FN): An undetected ground-truth bounding box;

Based on these definitions, the concepts of precision and recall can be formally expressed respectively as:

<!--- Precision Equation --->
<p align="center">
<img src="https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/data/images/precision_eq.png" align="center"/>
</p>
<!-- P \!\!\!&=&\!\!\!   \frac{\text{TP}}{\text{TP}+\text{FP}} = \frac{\text{TP}}{\text{all detections}}-->

<!--- Recall Equation --->
<p align="center">
<img src="https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/data/images/recall_eq.png" align="center"/>
</p>
<!-- R \!\!\!&=&\!\!\!
  \frac{\text{TP}}{\text{TP}+\text{FN}} = \frac{\text{TP}}{\text{all ground truths}} -->

The balance between precision and recall is considered by the average precision (AP) metric and its variations, as detailed below.

### Average Precision (AP)

The AP is a metric based on the area under the precision x recall curve and can be regarded as a trade-off between precision and recall for different confidence levels of the predicted bounding boxes. If the confidence of a detector is such that its FP is low, the precision will be high. However, in this case, many positives may be missed, yielding a high FN, and thus a low recall. Conversely, if one accepts more positives, the recall will increase, but the FP may also increase, reducing the precision.
In practice, a good object detector should find all ground-truth objects (FN=0 equivalent to a high recall), while identifying only relevant objects (FP=0 equivalent to a high precision).
Therefore, a particular object detector can be considered good if its precision stays high as its recall increases, which means that if the confidence threshold varies, both precision and recall remain high. Hence, a high area under the curve (AUC) tends to indicate both high precision and high recall. There are basically two approaches to evaluate the AUC: the **N-point interpolation** and **all-point interpolation**, as detailed next.

#### N-point interpolation

In the N-point interpolation, the shape of the precision x recall curve is summarized by averaging the maximum precision values at the set L, containing N equally spaced recall levels varying from 0 to 1, as given by:

<!--- N-point interpolation equation--->
<p align="center"> <a name="AP_Npointseq">
<img src="https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/data/images/AP_Npointseq.png" align="center"/>
</p>
<!-- {\rm AP} = \frac{1}{N} \sum_{R \in L} P_{\text{interp}}(R) -->

where,

<!--- P-interp equation --->
<p align="center">
<img src="https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/data/images/p_interp_eq.png" align="center"/>
</p>
<!-- P_{\text{interp}}(R) = \max_{\tilde{R}:\tilde{R} \geq R} P(\tilde{R}) -->

In this definition of AP, instead of using the precision P(R) observed at each recall level R, the AP 256 is obtained by considering the maximum precision Pinterp(R) whose recall value is greater than or equal to R. Popular applications of such interpolation method use N = 101 or N = 11.


#### All-point interpolation

In the all-point interpolation, instead of interpolating only N equally spaced points, one interpolates through all points in such way that

<!--- all-point-interpolation equation --->
<p align="center">  <a name="interpolating_all_points_part1">
<img src="https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/data/images/all_point_interpolation.png" align="center"/>
</p>
<!-- {\rm AP}_{\rm all} = \sum_{n} (R_{n+1} - R_n) P_{\text{interp}}(R_{n+1}) -->


<!--- continuation all-point-interpolation equation --->
<p align="center">
<img src="https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/data/images/cont_all_point_interpolation.png" align="center"/>
</p>
<!-- P_{\text{interp}}(R_{n+1}) = \max_{\tilde{R}:\tilde{R} \geq R_{n+1}}P(\tilde{R}) -->

In this case, instead of using the precision observed at only few points, the AP is now obtained by interpolating the precision at each level, taking the maximum precision whose recall value is greater than or equal to R_{n+1}.

### Mean average precision (mAP)

\subsubsection{Mean Average Precision}

Regardless the interpolation method, AP is obtained individually for each class. But in large datasets with many classes, it is also expected to have a unique metric able to represent the exactness of the detections among all classes.
For such cases, the mean AP (mAP) is applied, which is simply the average AP over all classes, that is:

<!--- mean average precision equation --->
<p align="center"> <a name="map_equation">
<img src="https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/data/images/map_eq.png" align="center"/>
</p>
<!--  {\rm mAP} = \frac{1}{C} {\sum_{i=1}^{C} {\rm AP}_{i}} -->

with $AP_{i}$ being the AP value in the $i$-th class and $C$ is the total number of classes being evaluated.

### Average recall (AR)

HERE: TODO
The average recall (AR) is another evaluation metric used to measure the assertiveness of object detectors proposed by~\cite{hosang2015makes} for a given class.
Instead of computing the recall at a particular IOU threshold, the AR computes the average recall at IOU thresholds from 0.5 to 1.
An IOU of 0.5 can be interpreted as a rough localization of an object and is the least acceptable IOU by most of the metrics.
An IOU equals to 1 is able to represent the perfect location of the detected object.
Therefore, by averaging recall values in the interval $[0.5, 1]$,
the model is evaluated for considerably accurate object location.

Let *o* be the IOU overlap between a ground truth and a detected bounding box
and $R(o)$ a function that retrieves the recall for a given IOU $o$.
The AR is defined as twice the AUC of the $R(o)$ and is given by

<!--- AR equation --->
<p align="center"> <a name="AR">
<img src="https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/data/images/AR.png" align="center"/>
</p>
<!--{\rm AR} = 2 \int_{0.5}^{1} R(o) \,do.-->


The [paper](https://arxiv.org/abs/1502.05082) that first presented the AR metric also gives a straightforward equation for the computation of the above integral from the discrete sample set, as twice the average of the excess IOU for all the ground-truths, that is,


<!--- AR_discrete equation --->
<p align="center">  <a name="AR_discrete">
<img src="https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/data/images/AR_discrete.png" align="center"/>
</p>
<!--{\rm AR} = \frac{2}{N} \sum_{n=1}^{N} \max({\rm IOU}_n - 0.5, 0)-->

where ${\rm IOU}_n$ denotes the best IOU obtained for a given ground truth $n$.

Interestingly, COCO also reports the AR, lthough its definition does not match  exactly that on Equation~\eqref{eq:AR_discrete}. Instead, what is reported as the AR is the average of the maximum obtained recall across several IOU thresholds,

<!--- AR_coco equation --->
<p align="center">  <a name="AR_coco">
<img src="https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/data/images/AR_coco.png" align="center"/>
</p>
<!-- {\rm AR} = \frac{1}{T} \sum_{t=1}^{T} \max_{r:P_t(r) > 0} r -->

where $P_t(r)$ is the precision for a given recall at the IOU threshold indexed by ** Effectively, a coarse approximation of the original integral is obtained.


### Mean average recall (mAR)
As the AR is calculated individually for each class, similarly to mAP, a unique AR value can be obtained considering the mean AR among all classes, that is:

<!--- mean average recall equation --->
<p align="center">
<img src="https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/data/images/mAR_eq.png" align="center"/>
</p>
<!-- {\rm mAR} = \frac{1}{C} {\sum_{i=1}^{C} {\rm AR}_{i}} -->


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

As previously explained, different methods can be applied to measure the AUC of the precision x recall curve. Considering the [N-point interpolation equation](#AP_Npointseq) to calculate the AP with N with *N=11*, the interpolation measures the recall in the points L=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], and to consider the all-point interpolation approach, let us consider the [All-point interpolation equation](#interpolating_all_points_part1). Both approaches result in different plots as shown below:


<!--- Interpolating curves --->
<p align="center">
<img src="https://github.com/rafaelpadilla/review_object_detection_metrics/blob/main/data/images/interpolations_toyexample.png" align="center"/>
</p>

When an IOU threshold *t=0.5* was applied (plots of the first row in image above), the 11-point interpolation method obtained *AP=88.64%* while the all-point interpolation method improved the results a little, reaching *AP=89.58%*. Similarly, for an IOU threshold of *t=0.75%* (plots of the second row in image above), the 11-point interpolation method obtained *AP=49.24%* and the all-point interpolation *AP=50.97%*.

In both cases, the all-point interpolation approach considers larger areas above the curve into the summation and consequently obtains higher results.
When a lower IOU threshold was considered, the AP was reduced drastically in both interpolation approaches. This is caused by the flexibility the threshold brings in considering TP detections.

Finally, if focus is shifted towards how well localized the detections are, it is sensible to consult the AR metrics. Computing twice the average excess IOU for the samples in this practical example as in [this AR equation](#AR_discrete) yields *AR=60%*, while computing the average max recall across the standard COCO IOU thresholds, that is *t={0.50, 0.55, ..., 0.95}*, as in [COCO AR equation](AR_coco), yields *AR= 66%*.
As the latter computation effectively does a coarser quantization of the IOU space, the values do diverge slightly.


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

This AP metric is widely used to evaluate detections in the PASCAL VOC dataset. It measures the AP of each class individually by computing the area under the precision x recall curve interpolating all points as presented in the [all-point interpolation equation](#interpolating_all_points_part1). In order to classify detections as TP or FP the IOU threshold is set to *t=0.5*.

### mAP with IOU Threshold *t=0.5*

This metric is also used by PASCAL VOC dataset and is calculated as the AP with IOU *t=0.5*, but the result obtained by each class is averaged as given in the [mAP equation](#map_equation).

### AP@.5 and AP@.75

These two metrics evaluate the precision x curve differently than the PASCAL VOC metrics. In this method, the interpolation is performed in *N=101* recall points,
with *L=[0, 0.01, ..., 1]*, as given in [N-point interpolation equation](#AP_Npointseq). Then, the computed results for each class are summed up and divided by the number of classes, as in the [mAP equation](#map_equation).

The only difference between AP@.5 and AP@.75 is the applied IOU thresholds. AP@.5 uses *t=0.5* whereas AP@.75 applies *t=0.75*. These metrics are commonly used to report detections performed in the COCO dataset.

### AP@[.5:.05:.95]

This metric expands the AP@.5 and AP@.75 metrics by computing the AP@ with ten different IOU thresholds (*t=[0.5, 0.55, ..., 0.95]*) and taking the average among all computed results.

### AP\textsubscript{S}, AP\textsubscript{M}, and AP\textsubscript{L}} \label{sec:AP_sizes}

These three metrics, also referred to as AP Across Scales, apply the AP@[.5,.05:.95] taking into consideration the size of the ground-truth object. AP\textsubscript{S} only evaluates the ground-truth objects of small sizes (area < $32^2$ pixels); AP\textsubscript{M} considers only ground-truth objects of medium sizes ($32^2$ < area < $96^2$ pixels); AP\textsubscript{L} considers large ground-truth objects (area > $96^2$) only.

When evaluating objects of a given size, objects of the other sizes (both ground-truth and predicted) are not considered in the evaluation. This metric is also part of the COCO evaluation dataset.

### AR\textsubscript{1}, AR\textsubscript{10}, and AR\textsubscript{100}

These AR variations apply the [AR equation](#AR_coco) with a limiting number of detections per image. Therefore, they calculate the AR given a fixed amount of detections per image, averaged over all classes and IOUs.
The IOUs used to measure the recall values are the same as in AP@[.5,.05:.95].

AR\textsubscript{1} considers up to one detection per image, while AR\textsubscript{10} and AR\textsubscript{100} consider at most 10 and 100 objects, respectively.

### AR\textsubscript{S}, AR\textsubscript{M} and AR\textsubscript{L}

Similarly to the AR variations with limited number of detections per image, these metrics evaluate detections considering the same areas as the AP Across Scales.


## **Spatio-Temporal Tube Average Precision (STT-AP)**
When dealing with videos, one may be interested in evaluating the model performance at  video level, i.e., whether the object was detected in the video as a whole. This metric is an extension of the AP metric that integrates spatial and temporal localizations; it is concise, yet expressive.
A spatio-temporal tube *To* of an object *o* is the spatio-temporal region defined as the concatenation of the bounding boxes of an object from each frame of a video, that is *T_o = [B<sub>o,q</sub> B<sub>o,q+1</sub> ... B<sub>o,q+Q-1</sub>*, where *B<sub>o,k</sub>* is the bounding box of the object *o* in frame *k* of the video that is constituted of *Q* frames indexed by *k= q, q+1,..., q+Q-1*.
Considering a ground-truth spatio-temporal tube *T<sub>gt</sub>* and a predicted spatio-temporal tube *T<sub>p</sub>*, the spatio-temporal tube IOU (STT-IOU) measures the ratio of the overlapping to the union of the "discrete volume" between *T<sub>gt</sub>* and *T<sub>p</sub>*, such that
$$ {\text{STT-IOU}} =\frac{\text{volume}(T_p \cap T_\textit{gt}\text{)}}{\text{volume}(T_p \cup T_\textit{gt})} = \frac{\displaystyle\sum_{k}\text{area of overlap in frame }k}{\displaystyle\sum_{k}\text{area of union in frame } k}, $$

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

Now activate the evironment

`conda activate <env_name>`

And you are ready to run!

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

##### annotation format
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

##### predictions format
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

##### run
```python
from src.evaluators.tube_evaluator import TubeEvaluator

tube_evaluator = TubeEvaluator(annot_filepath, preds_filepath)
res, mAP = tube_evaluator.evaluate(thr=0.5)
```

## Contributing

We appreciate all contributions. If you are planning to contribute with this repository, please do so without any further discussion.

If you plan to add new features, support other bounding box formats, create tutorials, please first open an issue and discuss the feature with us. If you send a PR without previous discussion, it might be rejected.

It is also important that for each new feature, supporting other bounding box formats, and metrics, a pytest must be created.

## References

