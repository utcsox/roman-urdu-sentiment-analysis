# roman-urdu-sentiment-analysis

This repository contains end-to-end code to automatically identify sentiment on a under-resource language (Roman-Urdu) on social media using machine learning.   

## Data
A data corpus comprising of more than 20000 records in Roman Udu (a limited resource language) was collected and tagged for Sentiment (Positive, Negative, Neutral).

Link to data: http://archive.ics.uci.edu/ml/datasets/Roman+Urdu+Data+Set

## Project Diretory Structure
```
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. 
│   ├── 1.0-cchen-exploratory-data-analysis                     
│   ├── 2.0_cchen_roman_urdu_ngram_models                     
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Functions to load data from roman urdo sentiment analysis dataset
│   │   └── make_dataset.py
│   │
│   ├── features       <- Functions to perform N-gram and sequence vectorization functions
│   │   └── vectorize_data.py
│   │
│   ├── models         <- Functions to build, train, and tune models with different hyperparameters
│   │   │                
│   │   ├── build_model.py
│   │   └── train_mlp_model.py
│   │   └── tune_mlp_model.py 
│   │ 
│   └── visualization  <- Functions to create exploratory and results oriented visualizations
│       └── visualize.py
```
