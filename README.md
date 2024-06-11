# Earthquake Impact Prediction Project

## Overview

This project aims to predict the impact of earthquakes using historical seismic data and soil bulk density information. Using various properties of earthquakes, such as magnitude, depth, location, and various geological features, in addition to soil bulk density, we evaluate various machine learning (ML) classifiers to determine which one is best suited to model the potential impact of an earthquake.  
We include soil bulk density in our analysis since the composition of the soil, besides the properties of an earthquake, has a large influence on the impact an earthquake has [^1].  
This project integrates data retrieval, preprocessing, and model evaluation to deliver predictive capabilities and insights into seismic hazards.

## Process

- **Data Retrieval**: Fetch earthquake data from a United States Geological Survey (USGS) rest API at https://earthquake.usgs.gov/fdsnws/event/1 and soil density data from a `.csv` file (`wosis_latest_bdwsod.csv`) downloaded from https://data.isric.org/geonetwork/srv/eng/catalog.search#/metadata/2f99e111-183c-11e9-aba8-a0481ca9e724.
- **Data Preprocessing**: Clean and prepare data for analysis, including handling missing values and feature engineering.  
We use the Modified Mercalli Intensity (mmi) scale[^2] to determine the impact an earthquake might have.
- **Exploratory Data Analysis (EDA)**: Exploration of the dataset to understand distributions, relationships, and patterns.
- **Predictive Modeling**: Implement and evaluate various machine learning classifiers using the balanced accuracy score as metric to select the best one to build a model to predict earthquake impacts.


## Classifiers Evaluated

- **Random Forest Calssifier (RFC)**: Captures complex interactions in the data and assesses feature importance, providing predictions by aggregating multiple decision trees.
- **Multinomial Logistic Regression**: Performs multinomial classification to separate data points into different classes.
- **Support Vector Machine (SVM)**: Handles complex classification tasks and ensures robust separation in high-dimensional spaces.
- **K Nearest Neighbors (KNN)**: Classifies data points based on the proximity of similar data points.
- **Decision Tree (DT)**: Classifies data using series of binary decisions organized into a hierarchical tree structure.

## Files
### Data Files 
 * `wosis_latest_bdwsod.csv` - Soil bulk density data file.
 * `earthquake_data.csv` - Raw dataset. File combines earthquake data retrieved from USGS and soil bulk density data.
 * `earthquake_data_standardized.csv` - Standardized/Scaled and reduced dataset used for analysis. Contains only features we will use to build our models. Based on `earthquake_data.csv`. 

### Main Program Files
This section lists the main program files used. Details are provided in another section below.
* `retrieve_data.ipynb` - Retrieves earthquake and soil data and creates `earthquake_data.csv`.
* `preprocess_data.ipynb` - Takes `earthquake_data.csv`, preprocesses the data and creates `earthquake_data_standardized.csv` as output.
* `evaluate_classifiers.ipynb` - Takes `earthquake_data_standardized.csv` as input and evaluates five different classifiers to determine which one is best suited to build our model.
* `earthquake_model.ipynb` - Uses pipelines to build the model to predict earthquake intensity.

### Exploratory Program and Data Files
* `lr_model_predictions.ipynb` - Preliminary evaluation of Mutinomial Logistic Regression Classifier.
* `lr_model_optimizations.ipynb`- Another preliminary evaluation of Mutinomial Logistic Regression Classifier.
* `svm_model_predictions.ipynb` - Preliminary evaluation of Support Vector Machines Classifier.
* `svm_model_optimization.ipynb` - Another preliminary evaluation of Support Vector Machines Classifier.
* `rfc_model_optimizations.ipynb` - Preliminary evaluation of Random Forest Classifier.
* `rfc_reduced_model_optimizations.ipynb` - Preliminary evaluation of Random Forest Classifier using reduced feature set.
* `knn_model_optimization.ipynb` - Preliminary evaluation of K-Nearest_neighbors Classifier.
* `PSHA_DSHA_model_predictions.ipynb` - Preliminary evaluation of Probabilistic Seismic Hazard Analysis (PSHA) and Deterministic Seismic Hazard Analysis (DSHA) models.
* `earthquake_data_reduced.csv` - A data file that contains the feature and target columns used to evaluate classifiers and build our model but before those features are scaled. This data file is no longer needed.

### Utility Program Files
* `optimization_utilities.py` - Used in `evaluate_classifiers.ipynb`. Utility functions for evaluating classifiers.
* `pipeline_utilities.py` - Used in `earthquake_model.ipynb`. Utility functions for preprocessing, splitting data, and building models in production pipeline.

### Reference Files
* README.md - Project overview and setup instructions
* EDA.md 

## Details of Data Collection and Preprocessing
### Retrieve Earthquake and Soil Data:
1. Retrieve Earthquake data using these filters 
   * Start date: 1/1/1995, end date: 12/31/2023.
   * Location: Contiguous United States as defined by minimum and maximum longitudes and latitudes.
   * Minimum earthquake mangintude: 3.0
2. Retrieve Soil Density Data (given by longitude and latitude) and filter to just US data using the same minimum and maximum longitudes and latitudes used to filter the earthquake data.
3. Calculate an average soil bulk density from the different soil layers given in the original data.
4. Combine earthquake and soil bulk density data by finding the soil bulk density record with the nearest longitude and latitude to the longitude and latitude of a given earthquake record. 
5. Save combined earthquake and soil bulk density data in the data file `earthquake_data.csv`.
### Data Pre-Processing
1. Retrieve and read earthquake data from `earthquake_data.csv`.
2. Check data types.
3. Drop rows with `NaN` values.
4. Assess the effect of dropping rows with `NaN` values by comparing the data distribution for each numerical feature before and after dropping rows with `NaN` values.  
**Result**: Distributions do not change substantially, so it is safe to drop rows with `NaN` values.
5. Since we are using `mmi` as our target feature, calculate correlations between `mmi` and all other numerical features.
6. Use correlation and feature definitions to identify features that are also metrics for the impact of an earthquake.
7. Drop columns identified in step 6 to prevent data leakage. Also, drop columns that are irrelevant.
8. Add a column with different `mmi` categories. We define three categories of earthquakes depending on their maximum `mmi` value.
   * **Weak**: Earthquakes with `mmi` score of less than 4.
   * **Moderate**: Earthquakes with an mmi score between 4 and less than 5.
   * **Strong**: Earthquakes with an mmi score of 5 or higher.
9. Check how many data points fall into each category to make sure categories contain roughly an equal number of data points.  
**Result**: Categories are reasonably balanced since
   * Weak category contains 352 data points.
   * Moderate category contains 353 data points.
   * Strong contains 252 data points.
10. Drop the original `mmi` column to prevent data leakage.
11. Scale all data using Standard Scaler.
12. Save data to csv `earthquake_data_standardized.csv`.
13. Run preliminary Random Forest Classifier model on scaled, reduced, and cleaned data to assess feasability of building a mopdel with the given data and determine which features might be dominant.  
**Result**: Balanced test accuracy score looks promising. However, model is overfitting. No set of features is dominant.

## Details of Model Evaluation and Tuning
We use the balanced accuracy score of the test data set as the metric to evaluate the performance of the different models.
During our initial exploration of the performance of different models, we noticed that the performance was quite sensitive to the values of the `random_state` seed variable. Since the selection of the best performing model should not depend on the randomly chosen `random_state` variable, we performed model evaluations and tuning with five different values (of 3, 7, 13, 29, 42) for the `random_state` variable. We then used the average of the balanced accuracy scores over the different values of the `random_state` variable to evaluate the performance of each model.  
We used the following classifiers to build models using the full feature set:
* Random Forest,
* K-Nearest_Neighbors,
* Decision Tree,
* Multinomial Logistic Regression, and
* Support Vector Machine.

For each model we split the data into train and test data sets and calculated average balanced accuracy scores for the train and test data sets as described above.  
Based on the balanced test accuracy score, we picked the three best performing models for further optimization and tuning. Those models were
* Random Forest - It had the best balanced test accuracy score (of 0.630) but with a balanced train accuracy score of 1.0 was overfitting the train data.
* Multinomial Logistic Regression - It had the second best balanced test accuracy score (of 0.594) and the smallest difference between balanced train and test accuracy scores (of 0.628 and 0.594, respectively) indicating the least overfitting of the train data.
* Support Vector Machine - It had the third best balanced test accuracy score (of 0.589) with a still relatively small difference between balanced train and test accuracy scores (of 0.737 and 0.589, respectively).
Not only are the balanced test accuracy scores of these models the best, but they are also within the variation we saw when picking different values for the `random_state` variable. Further reasons for discarding the Decison Tree and the KNN classifier are
* Low balanced test accuracy for both (0.525 for Decision Tree and 0.477 for KNN).
* A balanced train accuracy score of 1.0 for the Decision Tree indicating large overfitting of the train data for this model.

Next, we addressed the overfitting of the Random Forest Classifier model. A standard way to address overfitting is to reduce the number of features in the model. We used two approaches to find a set of features that would result in an acceptable (i.e., > 0.6) balanced accuracy score for the test data and a balanced accuracy score for the train data of substantially less than 1.0. Those approaches were
* P-values and
* PCA.
Again, we performed our analysis with the five different `random_state` seed values mentioned above.
### Result of P-Value Analysis
1. Depending on the seed value the size of the p-values and their order varied.
2. We had to consistently remove 29 out of 30 features for the balanced accuracy score for the test data to drop below 1.0.  
Therefore we concluded that using the p-values approach to reduce the overfitting of the Random Forest Classifier model was unsuccessful.
### Result of PCA
We varied the number of components for our PCA from 2 to the number of features minus 1 and did not find that the balanced accuracy for the train data dropped below 1.0.  
Therefore we concluded that using PCA to reduce the overfitting of the Random Forest Classifier model was unsuccessful as well.

Next, we turned to hyperparameter tuning. We performed hyperparameter tuning on all three of our best models.
### Result of Hyperparameter Tuning for Random Forest Classifier Model

### Result of Hyperparameter Tuning for Multinomial Logistic Regression Model

### Result of Hyperparameter Tuning for Support Vector Machine Model


## Conclusions
Present the results of the model and any conclusions drawn from the analysis.

## Future Work
Discuss any additional questions that surfaced and outline plans for future development.
### Future Considerations

- **Geospatial Analysis**: We plan to enhance our visualization capabilities by creating detailed maps that show earthquake occurrences and densities.
- **Seismic Hazard Analysis**: We aim to integrate both Probabilistic Seismic Hazard Analysis (PSHA) and Deterministic Seismic Hazard Analysis (DSHA) to evaluate potential earthquake impacts comprehensively.


## Authors
Pedro Zurita
Christoph Guenther
Ashwini Kumar

*******************************************************************************************
# Class Project Requirements 

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

## Footnotes
[^1]: Nolan, Joe (May 6, 2022). *The Effects of Soil Type on Earthquake Damage*, WSRB website, https://www1.wsrb.com/blog/the-effects-of-soil-type-on-earthquake-damage, accessed on 6/10/2024.
[^2]: Earthquake Hazards Program. *The Modified Mercalli Intensity Scale*, USGS website, https://www.usgs.gov/programs/earthquake-hazards/modified-mercalli-intensity-scale, accessed on 6/10/2024.

