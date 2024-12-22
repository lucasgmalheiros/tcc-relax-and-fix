import functions_ml as fml
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss, f1_score, jaccard_score, accuracy_score

# PARAMETERS
BINARY_CLASSIFICATON = False
TOLERANCE_LIMIT = 0.5 / 100

# 1. Get dataset
results = pd.read_csv('datasets/instances_results.csv')
features = pd.read_csv('datasets/multi_plant_instance_features.csv')
dataset = fml.create_dataset(features, results)

# 2. Create target columns for multi label classification
dataset = fml.create_multi_label_columns(dataset, TOLERANCE_LIMIT)

# 3. Train and test split
X_train, X_test, y_train, y_test = fml.train_test_split_multilabel(dataset, test_size=0.3, random_state=2112, label_prefix='RF_')

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
