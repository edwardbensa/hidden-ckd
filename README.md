# Hidden CKD
This project is part of a larger [study](https://www.hiddenckd.co.uk/) that aims to assess kidney health inequality in the UK and also shed more light on the increased risk that Black people have of developing chronic kidney disease ([CKD](https://www.nhs.uk/conditions/kidney-disease/)), a long-tern condition where the kidneys are significantly damaged and cannot perform essential functions including blood pressure regulation and waste removal.

Kidney Care UK [estimates](https://kidneycareuk.org/news-from-kidney-care-uk/one-in-four-people-unaware-of-the-main-signs-of-chronic-kidney-disease/) that 14.3% of people with CKD are unaware they have the condition. Usually, CKD is diagnosed using the estimated glomerular filtration rate (eGFR) or during albumin-to-creatinine ratio (uACR) tests. This analysis, however, seeks to determine if simpler tests can be used to identify individuals who may have early stage CKD using only basic tests and medical information such as ethnicity (Black and South Asian people are at higher risk), blood pressure, whether or not the patient has been diagnosed with diabetes (diabetics are at an increased risk of developing CKD), weight, and height.

## Project Organisation

```
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialised models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering)
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── data                
    │   ├── __init__.py 
    │   ├── clean_data.py       <- Scripts to clean data          
    │   └── prepropcessing.py   <- Code to create features and targets for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plotting                
        ├── __init__.py 
        └── plots.py            <- Code to create visualisations 
```

--------

