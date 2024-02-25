# 🚖 Taxi Demand Predictor 🚀

Welcome to the NYC Taxi Demand Prediction project! Here, we use advanced machine learning techniques to forecast the demand for taxis in New York City, enabling efficient fleet distribution and potential revenue increase. This guide provides an overview of the project, including its components, structure, and how to get started.

![Banner](https://github.com/SaiSpr/updated_taxi/assets/63905195/24c319e3-9fe2-49cc-9808-4d007936d6f9)


## Table of Contents

- [Web App Overview](#web-app-overview)
- [Feature/Training/Inference Pipelines](#fti-pipelines)
- [Code Structure](#code-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Contribute](#contribute) 
- [License](#license)

## Web App Overview 🌟

Our web application utilizes a trained LightGBM model to predict the demand for taxis in the upcoming hour. It fetches data from January 2022 onwards, stored in a Feature Store on HopsWorks, with new data fetched every 60 minutes using GitHub Actions.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://updatedtaxi-nmmahsydyuz9o2gxymewwd.streamlit.app/)

The web app provides a user-friendly interface to visualize real-time predictions through time series plots. Users can filter by location and monitor key performance metrics.

## Feature/Training/Inference Pipelines

Our project follows the FTI (Feature, Training, Inference) architecture, which offers a unified structure for both batch and real-time ML systems. Here's a breakdown of each pipeline:

![3-pipelines](https://github.com/SaiSpr/updated_taxi/assets/63905195/3541d380-232f-4863-9629-7bce59f1f18e)


### Feature Pipeline

- **11_backfill_feature_store.ipynb**: Downloads and consolidates raw taxi ride data, transforms it into a time-series format, and uploads it to a centralized feature store.
- **12_simulated_feature_pipeline.ipynb**: Fetches simulated recent production data, transforms it into a time-series format, and populates a feature store.

### Training Pipeline

- **13_model_training_pipeline.ipynb**: Establishes a training pipeline, creates feature views, transforms time-series data into features and targets, splits the data, and tunes a LightGBM model using Optuna.

### Inference Pipeline

- **14_inference_pipeline.ipynb**: Fetches recent data, predicts taxi demand using LightGBM, and saves predictions to a feature store.

## Code Structure

Our project maintains an organized directory structure:

```
.
├── README.md
│
├── raw
│   ├── rides_2022-01.parquet
│   ├── rides_2022-02.parquet
│   └── ...
│
├── transformed
│   ├── tabular_data.parquet
│   ├── ts_data_rides_2022_01.parquet
│   ├── validated_rides_2022_01.parquet
│   └── ...
│
├── models
│
├── notebooks
│   ├── 01_load_and_validate_raw_data.ipynb
│   ├── 02_transform_raw_data_into_ts_data.ipynb
│   ├── 03_transform_ts_data_into_features_and_targets.ipynb
│   └── ...
│
├── pyproject.toml
│
├── scripts
│
├── src
│
└── tests
```

## Installation

To set up the environment, clone the repository and install dependencies:

```shell
git clone https://github.com/SaiSpr/Taxi_Demand_Prediction/
cd taxi_demand_prediction
poetry install
poetry shell
```

Stay tuned for upcoming features and enhancements!
