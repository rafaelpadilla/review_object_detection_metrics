<div align="center" markdown>
<img src="https://i.imgur.com/PlwXDGP.png"/>

# Interactive Object Detection Metrics 

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#References">Acknowledgement</a>
  <a href="#References">Screenshot</a>
</p>


[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervise.ly/apps/supervisely-ecosystem/review_object_detection_metrics/supervisely)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervise.ly/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/review_object_detection_metrics)
[![views](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/review_object_detection_metrics/supervisely&counter=views&label=views)](https://supervise.ly)
[![used by teams](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/review_object_detection_metrics/supervisely&counter=downloads&label=used%20by%20teams)](https://supervise.ly)
[![runs](https://app.supervise.ly/public/api/v3/ecosystem.counters?repo=supervisely-ecosystem/review_object_detection_metrics/supervisely&counter=runs&label=runs&123)](https://supervise.ly)

</div>

# Overview
This app is an interactive visualization for object detection metrics. 
It allows to estimate performance of any detection model by comparing ground truth annotations with predictions.

# Acknowledgement
We forked the source code of [Object Detection Metrics repo](https://github.com/rafaelpadilla/Object-Detection-Metrics).
It is most popular implementation of detection metrics. We adopted it for Supervisely format and created the interactive 
dashboard with tons of visualizations. 

# How To Run
1. Prepare two projects: ground truth and predictions. Each bounding box in predictions project should have a tag 
`confidence` with value between 0 and 1. You can use our prepared sample projects: `XXX` and `XXX`
3. We prepared PascalVOC sample with predictions. Each box has tag `confidence`, Add project `XXX` from ecosystem.

# Screenshot
<img src="https://i.imgur.com/xBrUAv9.png"/>
