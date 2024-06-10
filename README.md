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

## Project Structure

# Project Requirements 

#### Data Model Implementation (25 points)

*   There is a Jupyter notebook that thoroughly describes the data extraction, cleaning, and transformation process, and the cleaned data is exported as CSV files for the machine learning model. (10 points)
    
*   A Python script initializes, trains, and evaluates a model or loads a pretrained model. (10 points)
    
*   The model demonstrates meaningful predictive power, at least 75% classification accuracy or 0.80 R-squared. (5 points)
    

#### Data Model Optimization (25 points)

*   The model optimization and evaluation process showing iterative changes made to the model and the resulting changes in model performance is documented in either a CSV/Excel table or in the Python script itself. (15 points)
    
*   Overall model performance is printed or displayed at the end of the script. (10 points)
    

#### GitHub Documentation (25 points)

*   GitHub repository is free of unnecessary files and folders and has an appropriate .gitignore in use. (10 points)
    
*   The README is customized as a polished presentation of the content of the project. (15 points)
    

#### Presentation Requirements (25 points)

Your presentation should cover the following:

*   An executive summary or overview of the project and project goals. (5 points)
    
*   An overview of the data collection, cleanup, and exploration processes. (5 points)
    
*   The approach that your group took in achieving the project goals. (5 points)
    
*   Any additional questions that surfaced, what your group might research next if more time was available, or share a plan for future development. (3 points)
    
*   The results and conclusions of the application or analysis. (3 points)
    
*   Slides effectively demonstrate the project. (2 points)
    
*   Slides are visually clean and professional. (2 points)

 # Project Title
 ## Earthquake Maginitude Predictions 

#### Executive Summary
Provide a brief overview of the project, including the main goals and objectives.

#### Data Collection and Cleaning
Explain the process of data collection and the steps taken to clean and transform the data.

#### Model Training and Evaluation
Describe the approach taken to train and evaluate the machine learning model. 

#### Model Optimization
Document the iterative process of model optimization and the resulting performance improvements.

#### Results and Conclusions
Present the results of the model and any conclusions drawn from the analysis.

#### Future Work
Discuss any additional questions that surfaced and outline plans for future development.

#### Installation
Provide instructions for setting up the project locally.

#### Usage
Explain how to run the code and what to expect from the outputs.

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



