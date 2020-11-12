Review on Object Detection Metrics
==============================

Repositório de trabalho para o journal paper [Electronics - Special Issue](https://www.mdpi.com/journal/electronics/special_issues/learning_based_detection).  

**Deadline da implementação: 18 de novembro**

### Tarefas:

#### Wesley:  

i) Implementação da métrica proposta para detecção de objetos em vídeos (tubos)  

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



