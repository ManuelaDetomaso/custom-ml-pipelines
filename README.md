custom-ml-pipelines
==============================

Different apporaches to building ML pipelines

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── input          <- Input/raw data
    │   ├── databolt_output<- Intermediate data that has been transformed.
    │   └── output         <- Processed files, partitoned data and model artifacts
    |       ├──model_artifacts
    |       └── partitions
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- training and predictions workflows
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-md-custom-ml-pipeline`.
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
    │   ├── models         <- Scripts to create training pipelines to get models and then use 
    |   |                     trained models to make predictions
    │   │   │   
    |   |   ├── sklearn_predict_model.py
    │   │   ├── sklearn_train_model.py
    |   |   ├── databolt_predict_model.py
    │   │   ├── databolt_train_model.py
    │   │   ├── custom_predict_model.py
    │   │   └── custom_train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
