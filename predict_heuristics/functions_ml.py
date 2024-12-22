import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss, f1_score, jaccard_score, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from xgboost import XGBClassifier
logging.basicConfig(level=logging.INFO)

def create_dataset(dataset_features: pd.DataFrame, dataset_results: pd.DataFrame) -> pd.DataFrame:
    """Merge results and features dataset"""
    # Filter for results columns
    dataset_results = dataset_results[['Instancias'] + [col for col in dataset_results.columns if col.startswith('Obj_') or col.startswith('Time_')]]
    # Remove infeasible instances from results
    dataset_results = dataset_results[dataset_results['Obj_RF_T_0'] != np.inf]
    # Merge features and results (inner join to remove infeasibles)
    data = dataset_features.merge(dataset_results, left_on='instance', right_on='Instancias', how='inner').drop(columns=['Instancias', 'instance'])
    # Shuffle data
    data = data.sample(frac=1, random_state=2112)
    return data


def create_target(data: pd.DataFrame, tolerance: float = 0.01, time_limit: float = 0.99 * 1800, binary_classification: bool = False) -> pd.DataFrame:
    """Define a single target for each entry in dataset based on the lowest objective and quickest time within a tolerance level."""
    objective_columns = [col for col in data.columns if col.startswith('Obj_')]
    time_columns = [col for col in data.columns if col.startswith('Time_')]
    # Prepare to calculate the best method per instance
    data['TARGET'] = None
    # Iterate over each row to determine the best method according to the defined methodology
    for index, row in data.iterrows():
        # Get the minimum objective function value
        objective_values = row[objective_columns]
        best_obj = objective_values.min()
        best_method = objective_values.idxmin()  # Index of method with the minimum objective value
        # Calculate deviation and filter methods within the acceptable tolerance
        deviations = (objective_values - best_obj) / best_obj
        equivalent_methods = deviations[deviations <= tolerance].index.tolist()
        # Select the method with the shortest time within the tolerance range
        corresponding_times = ['Time_' + method.replace('Obj_', '') for method in equivalent_methods]
        times = row[corresponding_times]
        # Check if any method finishes before the time limit
        valid_times = times[times < time_limit]
        if not valid_times.empty:
            # Get the method with the minimum time
            fastest_method = valid_times.idxmin()
            data.at[index, 'TARGET'] = fastest_method.replace('Time_', '')
            data.at[index, 'TARGET_TIME'] = row[fastest_method]
        else:  # If there are no faster method, keep the one with the minimum objective value as target
            data.at[index, 'TARGET'] = best_method.replace('Obj_', '')
            data.at[index, 'TARGET_TIME'] = row['Time_' + best_method.replace('Obj_', '')]
    # For binary classification, the target can only be Gurobi or RF. Choose this for higher level analysis
    if binary_classification:
        data['TARGET'] = data['TARGET'].apply(lambda x: 'Gurobi' if x == 'RF_T_0' else 'RF')
    # Remove results columns
    data = data.drop(columns=objective_columns + time_columns)
    return data


def create_multi_label_columns(data: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """
    Create multi-label binary columns for each method based on a tolerance limit.
    Each method will have a column, with 1 indicating the method is within the tolerance and 0 otherwise.
    """
    # Identify the objective function columns (methods)
    objective_columns = [col for col in data.columns if col.startswith('Obj_')]
    time_columns = [col for col in data.columns if col.startswith('Time_')]
    # Initialize multi-label binary columns for each method
    for col in objective_columns:
        method_name = col.replace('Obj_', '')  # Extract method name from the column
        data[method_name] = 0  # Create a column for each method with default value 0
    # Iterate over each row to determine the equivalent methods within tolerance
    for index, row in data.iterrows():
        # Get the minimum objective function value
        objective_values = row[objective_columns]
        best_obj = objective_values.min()
        # Calculate deviations and find methods within the acceptable tolerance
        deviations = (objective_values - best_obj) / best_obj
        equivalent_methods = deviations[deviations <= tolerance].index.tolist()
        # Update the corresponding columns for equivalent methods
        for method in equivalent_methods:
            method_name = method.replace('Obj_', '')  # Extract method name from the column
            data.at[index, method_name] = 1  # Set the corresponding method column to 1
    # Remove results columns
    data = data.drop(columns=objective_columns + time_columns)
    return data


def train_test_split_multilabel(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42, label_prefix: str = 'RF_'):
    """Automatically splits the data into training and testing sets for multi-label classification."""
    label_columns = [col for col in data.columns if col.startswith(label_prefix)]
    feature_columns = [col for col in data.columns if col not in label_columns]
    # Separate features and labels
    X = data[feature_columns]
    Y = data[label_columns]
    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # PARAMETERS
    BINARY_CLASSIFICATON = False
    TOLERANCE_LIMIT = 0.005

    # 1. Get dataset
    logging.info('Creating complete dataset...')
    results = pd.read_csv('datasets/instances_results.csv')
    features = pd.read_csv('datasets/multi_plant_instance_features.csv')
    dataset = create_dataset(features, results)
    logging.info('Dataset processed successfully!')

    # 2. Create target columns for multi label classification
    dataset = create_multi_label_columns(dataset, TOLERANCE_LIMIT)

    # 3. Train and test split
    X_train, X_test, y_train, y_test = train_test_split_multilabel(dataset, test_size=0.2, random_state=2112, label_prefix='RF_')

    # 4. Creating the MultiOutput Classifier
    classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=2112))

    # 5. Fitting the classifier on the training data
    classifier.fit(X_train, y_train)

    # 6. Making predictions on the test set
    predictions = classifier.predict(X_test)

    # 7. Evaluate the model
    hamming = hamming_loss(y_test, predictions)
    print("Hamming Loss:", hamming)

    f1 = f1_score(y_test, predictions, average='micro')
    print("Micro-Averaged F1 Score:", f1)

    jaccard = jaccard_score(y_test, predictions, average='samples')
    print("Jaccard Similarity Score:", jaccard)

    subset_accuracy = accuracy_score(y_test, predictions)
    print("Subset Accuracy:", subset_accuracy)
