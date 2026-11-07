# RoadWatch – Accident Hotspot Analysis and Severity Prediction

**RoadWatch** is a road-safety analytics and accident severity prediction project focused on Indian road accident data. It combines accident data preprocessing, machine-learning-based severity classification and Bengaluru traffic accident analysis to support data-driven road-safety insights.

## Project Overview

Road accidents are influenced by several factors, including road conditions, weather, vehicle types, accident locations and time-related patterns. Understanding these factors can help identify areas requiring further investigation and improve road-safety planning.

RoadWatch aims to analyse available Indian accident datasets, identify patterns in recorded accidents and develop machine-learning models to classify accident severity.

The current implementation focuses on two data sources:

- **Indian highway accident dataset:** Individual accident records containing accident severity and contextual information.
- **Bengaluru Traffic Police data:** Police-station-level accident and fatality aggregates used for temporal and geographical analysis.

## Main Features

### 1. Accident Data Preprocessing

- Data cleaning and validation.
- Removal of exact duplicate records.
- Handling of missing and invalid values.
- Standardisation of supported categorical variables.
- Date and time feature extraction.
- Preparation of datasets for analysis and machine-learning experiments.

### 2. Accident Severity Classification

RoadWatch includes a binary machine-learning classification experiment that groups accident severity into two categories:

| Category | Original severity classes |
|---|---|
| Severe | Fatal and Grievous Injury |
| Non-severe | Minor Injury and No Injury |

The experiment compares multiple machine-learning algorithms:

- Extra Trees
- Random Forest
- CatBoost
- XGBoost
- Logistic Regression

A chronological train-validation-test strategy is used. Model selection and decision-threshold selection are performed using validation data, while the final test partition is held out for evaluation.

**Selected model:** Extra Trees Classifier

**Selected decision threshold:** 0.43

#### Final Severity V2 Test Results

| Metric | Result |
|---|---:|
| Accuracy | 57.46% |
| Balanced Accuracy | 58.43% |
| Severe Precision | 49.75% |
| Severe Recall | 64.71% |
| Severe F1-score | 56.25% |
| Severe F2-score | 61.04% |
| Non-severe Specificity | 52.15% |
| ROC-AUC | 0.5985 |

The binary model identified 99 of the 153 severe cases in the held-out test dataset.

A four-class Extra Trees reference experiment was also conducted, achieving 30.11% accuracy and a macro F1-score of 24.56%.

These results represent a research baseline. The model has limited discriminatory capability and should not be considered suitable for operational safety decisions.

### 3. Bengaluru Accident Analysis

The Bengaluru Traffic Police dataset provides station-level accident and fatality aggregates.

The project uses this data to support analysis such as:

- Year-wise accident trends.
- Police-station-level comparisons.
- Fatality and non-fatality patterns.
- Identification of stations with comparatively higher recorded accident counts.

The available data is aggregated by police station and year. It should not be interpreted as precise accident coordinates or individual crash locations.

### 4. Dashboard and Backend

RoadWatch uses a React-based frontend and a Python backend built with FastAPI.

The application is intended to bring together accident analytics, severity prediction and relevant visualisations through a single interface.

The dashboard should only display results supported by the available datasets and implemented backend functionality.

## Datasets

### Indian Highway Accident Dataset

Contains individual accident records with contextual variables such as:

- Accident severity.
- Accident location classification.
- Road features and road conditions.
- Weather conditions.
- Vehicle types involved.
- Accident date and time.
- Chainage-related information where available.

**Data limitation:** The available accident records do not provide reliable GPS coordinates or corridor identifiers for all observations. Consequently, accurate geographical clustering across highway routes cannot currently be established from this dataset alone.

### Bengaluru Traffic Police Dataset

Contains aggregated police-station-level accident statistics across multiple years.

The data supports station-level and temporal comparisons. It does not provide the individual crash coordinates needed for precise point-based hotspot mapping.

## Machine Learning Methodology

The Severity V2 experiment follows these steps:

1. Load and preprocess accident records.
2. Map the original severity classes into severe and non-severe categories.
3. Extract relevant accident-context features.
4. Divide the records chronologically into training, validation and test partitions.
5. Train and compare candidate classification algorithms.
6. Select the model and classification threshold using validation performance.
7. Evaluate the selected model on the held-out test dataset.
8. Save the evaluation metrics and trained model artifacts.

The `Causes_D` variable is excluded from the V2 experiment because it may contain information recorded after an accident and could introduce target leakage.

## Dataset and Modelling Limitations

The current implementation has several important limitations:

- The highway dataset contains recorded accidents rather than both crash and non-crash observations.
- Reliable GPS coordinates and corridor identifiers are unavailable for the required highway spatial analysis.
- Accident counts alone do not provide a traffic-exposure denominator.
- The severity classifier predicts injury severity among recorded accidents, not the probability of a future accident occurring.
- The Bengaluru data is aggregated by police station and year.
- The severity model's predicted probabilities have not been calibrated.
- Additional information such as traffic volume, vehicle speed, road geometry and more detailed geographical data would improve future analysis.

These limitations are considered when interpreting model results and dashboard outputs.

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | React |
| Backend | Python, FastAPI |
| Data processing | Python, Pandas, NumPy |
| Machine learning | Scikit-learn, Extra Trees, Random Forest |
| Additional model experiments | XGBoost, CatBoost |
| Model evaluation | Scikit-learn metrics |
| Model persistence | Joblib |
| Version control | Git and GitHub |

The listed technologies describe the project stack and experiment dependencies; individual libraries should only be considered part of the active execution pipeline where they are actually used.

## Project Structure

The repository contains the frontend, backend, data-processing code, machine-learning experiments and supporting project documentation.

The exact directory structure may vary depending on the current development branch.

Important Severity V2 output files include:

- `highway_severity_v2_metrics.json`
- `highway_severity_v2_validation_comparison.csv`
- `highway_severity_v2_binary_best.joblib`
- `highway_severity_v2_four_class_reference.joblib`

These files are generated by running the Severity V2 experiment.

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/Akshayaraju1409/Accident-Hotspot-Prediction.git
cd Accident-Hotspot-Prediction
```

### 2. Set up the Python environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the backend dependencies:

```bash
pip install -r requirements.txt
```

### 3. Start the backend

Run the FastAPI application using the actual backend entry point configured in the repository.

For example, if the entry point is `main.py` and the FastAPI instance is named `app`:

```bash
uvicorn main:app --reload
```

### 4. Start the frontend

Navigate to the frontend directory:

```bash
cd frontend
npm install
npm run dev
```

Use the frontend development URL printed in the terminal to access the dashboard.

**Note:** Confirm the actual backend entry point, frontend directory and available scripts in the repository before running these commands.

## Future Improvements

Potential future enhancements include:

- Obtaining reliable geographical coordinates and highway corridor identifiers.
- Developing route-aware spatial hotspot analysis.
- Incorporating traffic volume and road-exposure information.
- Improving severe accident detection and reducing false-positive predictions.
- Evaluating additional machine-learning algorithms.
- Calibrating model probabilities.
- Improving the integration of Bengaluru station-level analytics with the dashboard.
- Expanding model evaluation using additional real-world datasets.

## Project Scope

RoadWatch is an academic research and development project intended to explore accident data analysis and machine-learning-based severity classification.

Its current results should be interpreted within the limitations of the available datasets. It is not an operational crash prediction, emergency response or road-safety decision system.