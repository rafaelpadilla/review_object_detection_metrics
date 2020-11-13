Review on Object Detection Metrics
==============================

Repositório de trabalho para o journal paper [Electronics - Special Issue](https://www.mdpi.com/journal/electronics/special_issues/learning_based_detection).  

**Deadline da implementação: 18 de novembro**

### Tarefas:

#### Wesley:  

i) Implementação da métrica proposta para detecção de objetos em vídeos (tubos)  
ii) Implementaço da métrica min-min-max usada no ImageNet challenge (detalhes [aqui](https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/evaluation))  

#### Thadeu:  

i) Implementação das métricas COCO  
ii) Implementação da métrica Average Delay  

Referências:  
[Métricas COCO](https://cocodataset.org/#detection-eval)  
[Métricas em geral](https://blog.zenggyu.com/en/post/2018-12-16/an-introduction-to-evaluation-metrics-for-object-detection/#fn3)  
[Average Recall](https://manalelaidouni.github.io/manalelaidouni.github.io/Evaluating-Object-Detection-Models-Guide-to-Performance-Metrics.html#average-recall-ar) 
[Paper Average Delay](https://arxiv.org/pdf/1908.06368.pdf)
[Implementação Average Delay](https://github.com/RalphMao/VMetrics)

Implementações:  
[Oficial COCO (cocoeval.py)](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)  
[Official Detectron (voc_eval.py)](https://github.com/facebookresearch/Detectron/blob/cbb0236dfdc17790658c146837215d2728e6fadd/detectron/datasets/voc_eval.py)  
[Implementação do paper de Niteroi](https://github.com/rafaelpadilla/Object-Detection-Metrics)  

#### Padilla:  

i) Escrever o paper  
ii) Codar módulo para reconhecer diferentes tipos de anotações  
iii) Desenvolver UI para facilitar o uso da API  

Referências:  
[Métricas para detecço de silhueta (mAP e mIOU)](https://www.youtube.com/watch?v=pDhCbYc0NBQ)  
[MOTP and MOTA](https://arxiv.org/pdf/2007.14863.pdf)  
[Evaluating image segmentation](https://www.jeremyjordan.me/evaluating-image-segmentation-models/) 
[Forum 1](https://stats.stackexchange.com/questions/462279/why-is-map-mean-average-precision-used-for-instance-segmentation-tasks)  
[Panoptic Segmentation Metric](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.pdf)  
[Pixelwise Instance Segmentation with a Dynamically Instantiated Network]()  
[Metrics for tracking](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4479472&casa_token=qVqK8NIQsNYAAAAA:F0uihc_37NUlyDWny3Yvwowb7k5xSM9ZZa7g8W5kAHVs0fXovPxNfQxpWNgPWBezt0MueFqzGA&tag=1)  
[Performance Evaluation of Object Detection Algorithms](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1048198&casa_token=7g5QwzVvBycAAAAA:3jQBF9mrWJ9OIYHO9O5gbvJme9q7nSNyRO7IJNJywuZCiliGOSkIiXpqrp6JiSpaHPv-fYnY3Q)  
[A Review of Detection and Tracking of Object from Image and Video Sequences](http://www.ripublication.com/ijcir17/ijcirv13n5_07.pdf)  
[A Review on Moving Object Detection and Tracking Methods in Video](https://acadpubl.eu/jsi/2018-118-16-17/articles/16/33.pdf)  
[Framework for Performance Evaluation of Face, Text, and Vehicle Detection and Tracking in Video: Data, Metrics, and Protocol](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4479472&casa_token=qVqK8NIQsNYAAAAA:F0uihc_37NUlyDWny3Yvwowb7k5xSM9ZZa7g8W5kAHVs0fXovPxNfQxpWNgPWBezt0MueFqzGA&tag=1)  
[Object Detection With Deep Learning: A Review](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8627998&casa_token=AQl_UN40niwAAAAA:yxPx_j_ul-lgCnon8F5FmHhRIkZJMNugSximoi6SHmLrG_W8l-UOb5YxvoTQ69HCdluwVJhrHQ)  
[New trends on moving object detection in video images captured by a moving camera: A survey]()  



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials. 
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io



