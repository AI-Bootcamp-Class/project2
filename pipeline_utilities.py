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

    num_rows_original = len(df_earthquake)
    # Drop rows with NaN and reset index
    df_cleaned = df_earthquake.dropna().reset_index(drop=True)
    num_rows_dropped_na = len(df_cleaned)

    percent_dropped = (num_rows_original-num_rows_dropped_na) / num_rows_original * 100
    print(f"Dropped {percent_dropped:.2f}% of rows with NaN values.")
    print(f"There are {num_rows_dropped_na} rows remaining.")
    print("="*100 + "\n")

    # Add a column for the 'mmi' class. This will be the target
    df_cleaned['mmi_class'] = [0 if mmi<4 else 1 if mmi>=4 and mmi<5 else 2 for mmi in df_cleaned['mmi']]

    # Drop columns 'id', 'time', 'place', 'felt', 'cdi', 'mmi', and 'significance'.
    # The columns 'id', 'time', and 'place' are not relevant for the model.
    # The columns 'felt', 'cdi', 'mmi', and 'significance' might introduce data leakage.
    columns_to_drop = ['id', 'time', 'place', 'felt', 'cdi', 'mmi', 'significance']
    df_final = df_cleaned.drop(columns=columns_to_drop)
    print(f"DataFrame to build model:")
    print(df_final.head())
    rows, cols = df_final.shape
    print(f"Number of rows: {rows}.")
    print(f"Number of columns: {cols}.")
    print("="*100 + "\n")

    # Define X and y
    X = df_final.drop(columns='mmi_class', axis=1)
    y = df_final['mmi_class']

    X_rows, X_cols = X.shape
    print(f"X:\n{X.head()}")
    print(f"Number of rows in X: {X_rows}.")
    print(f"Number of columns in X: {X_cols}.")
    print("-"*60)
    print(f"y:\n{y.head()}")
    print(f"Number of elements in y: {len(y)}.")
    print("="*100 + "\n")

    return(train_test_split(X, y))

def build_earthquake_model(df_earthquake):
    """
    Builds a model to predict the intensity of an earthquake using the 
    Modified Mercalli Intensity (MMI) Scale using the following steps:
    1. Preprocess the data,
    2. Split the data into train and test data sets,
    3. Scale the data using Standard Scaler,
    4. Fit a Random Forest Classifier model using optimized hyperparameters, and
    5. Calculate and print balanced accuracy scores.

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

    # Calculate balanced accuracy scores
    train_accuracy = calc_accuracy(X_train, y_train, pipeline)
    test_accuracy = calc_accuracy(X_test, y_test, pipeline)

    # Print balanced accuracy scores
    print(f"Balanced Train Accuracy Score: {train_accuracy:.3f}.")
    print(f"Balanced Test Accuracy Score: {test_accuracy:.3f}.")

    return(pipeline)

if __name__ == "__main__":
    print("This script should not be run directly! Import these functions for use in another file.")