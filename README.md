# Earthquake Impact Prediction Project

## Overview

This project aims to predict the impact of earthquakes using historical seismic data and soil bulk density information. By analyzing various factors such as magnitude, depth, location, and geological features, we employ machine learning models to forecast potential earthquake impacts and visualize earthquake occurrences. The project integrates data retrieval, preprocessing, and model evaluation to deliver robust predictive capabilities and insights into seismic hazards.

## Features

- **Data Retrieval**: Fetches earthquake data from USGS and soil density data from local files.
- **Data Preprocessing**: Cleans and prepares data for analysis, including handling missing values and feature engineering.
- **Exploratory Data Analysis (EDA)**: Comprehensive exploration of the dataset to understand distributions, relationships, and patterns.
- **Predictive Modeling**: Implements and evaluates various machine learning models to predict earthquake impacts.


## Key Models

- **Random Forest**: Captures complex interactions in the data and assesses feature importance, providing robust predictions by aggregating multiple decision trees.
- **Logistic Regression**: Performs trend analysis and binary classification to predict earthquake intensities.
- **Support Vector Machine (SVM)**: Handles complex classification tasks and ensures robust separation in high-dimensional spaces.
- **K Nearest Neighbors (KNN)**: Classifies earthquake impacts based on the proximity of similar data points.

## Future Considerations

- **Geospatial Analysis**: We plan to enhance our visualization capabilities by creating detailed maps that show earthquake occurrences and densities.
- **Seismic Hazard Analysis**: We aim to integrate both Probabilistic Seismic Hazard Analysis (PSHA) and Deterministic Seismic Hazard Analysis (DSHA) to evaluate potential earthquake impacts comprehensively.

## Project Structure 
 * earthquake_data_standardized.csv - Standardized dataset used for analysis
 * earthquake_data_reduced.csv - reduced dataset 
 * earthquake_data.csv - raw earthquake dataset
 * wosis_latest_bdwsod.csv - # Soil bulk density data file

### Files
*  retrieve_data.ipynb - # Retrieves earthquake and soil data
*  preprocess_data.ipynb - # Preprocesses the data
*  evaluate_classifiers.ipynb - # Evaluates and tunes classifiers
*  earthquake_model.ipynb -# Builds the model to predict earthquake intensity
* lr_model_predictions.ipynb - # Logistic regression model predictions
* svn_model_predictions.ipynb - # SVM model predictions
* rfc_model_optimizations.ipynb -# Random forest model predictions
### Scripts/
* optimization_utilities.py - # Utility functions for evaluating classifiers
*  pipeline_utilities.py - # Functions for preprocessing, splitting data, and building models
### Reference Files
* README.md - # Project overview and setup instructions
* EDA.md 

#### Data Collection and Cleaning
##### Data Pre-Processing
    1. Retrieve and read Earthquake data
    2. Check data types and filter our NaN values
    3. Target feature choosen is MMI
    4. Standard Scaler applied to data 
    5. Save Starndardized data to csv "earthquake_data_standardized.csv"
##### Retrieve Earthquake and Soil Data:
    1. Retrieve Earthquake data and define the specific columns and features needed
    2. Retrieve Soil Density Data and filter to just US data
    3. Find nearest lat and long soil density data points that correlate with the Earthquake data. 
    4. Merge Earthquake and Soil Density data and save it under file "earthquake_data.csv"
##### Model Training and Evaluation
    1. Run data set through KNN, RFC, SVN and Linear Regression Models
    2. Analysis Accuracy for Test and Train data sets
    3. Determine which models have optimal predictions 
#### Model Optimization
    1. Using Hyperparameters optimize the models
    2. Using P-Values to opimize models
    3. Use PCA values to opimize the models

#### Results and Conclusions
Present the results of the model and any conclusions drawn from the analysis.

#### Future Work
Discuss any additional questions that surfaced and outline plans for future development.

#### Authors
Pedro Zurita
Christoph Guenther
Ashwini Kumar

************************************************************************
# PROJECT PLAN

#### Setup Phase (Week 1: May 21 - May 27)
- **Day 1-2**: Establish project scope and objectives; setup GitHub repository and project documentation.
- **Day 3-5**: Research and select data sources, begin data extraction and initial cleaning.
- **Day 6-7**: Start initial model development and basic training.

#### Development and Optimization Phase (Week 2: May 28 - June 3)
- **Day 1-3**: Complete data cleaning and preprocessing; finalize initial model training.
- **Day 4-5**: Begin model optimization; implement different configurations and record changes.
- **Day 6-7**: Start drafting README and begin creating presentation slides.

##### Finalization and Review Phase (Week 3: June 4 - June 10)
- **Day 1-2**: Complete model optimization and ensure performance meets criteria.
- **Day 3-4**: Finalize all documentation (GitHub README, code comments).
- **Day 5-6**: Finalize presentation slides and rehearse delivery.
- **Day 7**: Conduct a full project review with all team members.

#### Closing and Presentation (June 11)
- Deliver the final presentation.
- Push final changes to GitHub.
- Gather feedback and discuss potential future developments.



