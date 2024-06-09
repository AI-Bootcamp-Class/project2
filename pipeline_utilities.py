import pandas as pd
from sklearn.model_selection import train_test_split
from optimization_utilities import calc_accuracy
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def preprocess_data(df_earthquake):
    """
    Preprocess earthquake data as follows:
    1. Drop rows with NaN values.
    2. Add a target column for the 'mmi' class.
       This is derived from the existing 'mmi' column as follows:
        * mmi_class = 0 if mmi < 4,
        * mmi_class = 1 if 4 <= mmi < 5
        * mmi_class = 2 if mmi >= 5
    3. Drop columns 'id', 'time', 'place', 'felt', 'cdi', 'mmi', and 'significance'.
    4. Create X and y
    5. Split data into train and test data sets.
    """

    # Drop rows with NaN and reset index
    df_cleaned = df_earthquake.dropna().reset_index(drop=True)

    # Add a column for the 'mmi' class
    df_cleaned['mmi_class'] = [0 if mmi<4 else 1 if mmi>=4 and mmi<5 else 2 for mmi in df_cleaned['mmi']]

    # Drop columns 'id', 'time', 'place', 'felt', 'cdi', and 'significance'
    columns_to_drop = ['id', 'time', 'place', 'felt', 'cdi', 'mmi', 'significance']
    df_final = df_cleaned.drop(columns=columns_to_drop)

    # Define X and y
    X = df_final.drop(columns='mmi_class', axis=1)
    y = df_final['mmi_class']

    return(train_test_split(X, y))

def build_earthquake_model(df_earthquake):
    """
    Builds a model to predict the intensity of an earthquake using the 
    Modified Mercalli Intensity (MMI) Scale using the following steps:
    1. Preprocess the data,
    2. Split the data into train and test data sets,
    3. Scale the data using Standard Scaler,
    4. Fit a Random Forest Classifier model using optimized hyperparameters, and
    5. Calculate balanced accuracy scores.

    Returns the trained model.
    """

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df_earthquake)

    # Create a list of steps for a pipeline that will scale the data and use a Random Forest
    # Classifier model.
    steps = [("Scale", StandardScaler()),
             ("Random Forest Classifier", RandomForestClassifier(max_depth=6))]

    # Create a pipeline object
    pipeline = Pipeline(steps)

    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Calculate balanced accuracy score
    train_accuracy = calc_accuracy(X_train, y_train, pipeline)
    test_accuracy = calc_accuracy(X_test, y_test, pipeline)

    print(f"Balanced Train Accuracy Score: {train_accuracy:.3f}.")
    print(f"Balanced Test Accuracy Score: {test_accuracy:.3f}.")

    return(pipeline)

if __name__ == "__main__":
    print("This script should not be run directly! Import these functions for use in another file.")