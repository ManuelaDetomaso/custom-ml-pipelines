custom-ml-pipelines
==============================

This repo explores different apporaches to building ML pipelines:

    - Tipical Scikit-Learn Pipeline.
    - Complete Custom pipeline not relying any specific packages for building workflows.
    - Databolt/Luigi pipeline based on d6tflow workflow structure.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── conf               <- Folder hosting the configurations stored in the config.json
    ├── data
    │   ├── input          <- Input/raw data
    │   ├── processed      <- Databolt processed files
    │   └── output         <- other pipelines' output files and model artifacts
    |       ├──model_artifacts
    |       └── partitions
    │
    ├── notebooks          <- Jupyter notebooks with examples of how to use alternative pipeline structures
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    |   ├── config.py
    │   ├── input_collector.py
    │   ├── custom_pipeline_modeler.py
    |   ├── databolt_pipeline_modeler.py
    │   └── sklearn_pipeline_modeler.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
