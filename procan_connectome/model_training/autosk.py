import autosklearn.classification
import autosklearn.metrics
from procan_connectome.config import DATA_PATH, RANDOM_STATE, LOGGER_LEVEL
import pandas as pd 
import numpy as np 
import os
from sklearn.model_selection import train_test_split
import datetime
from sklearn.metrics import accuracy_score
import pickle
from procan_connectome.utils.load_dataset import get_rf_dataset



NOW = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}"
OUTPUT_FOLDER = os.path.join(DATA_PATH, 'autosklearn_output',NOW+'_important_features_autosklearn_output')


# df = pd.read_csv(os.path.join(DATA_PATH, "combined_datasets.csv"))
# df = df.set_index('ID')
df = get_rf_dataset(threshold=0.001)
X, y = df.drop(columns='label'), df['label']
X, y = df.drop(columns='label'), df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y, 
    test_size=0.33,
    random_state=RANDOM_STATE
)

print(f"Loaded X_train and X_test with respective shapes: {X_train.shape} and {X_test.shape}")

classifier = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=86400,  #Set to 43200 for 12 hours
    seed = RANDOM_STATE, 
    output_folder=os.path.join(OUTPUT_FOLDER),
    memory_limit=14000,  # Set to 20000 for 20GB
    delete_output_folder_after_terminate=False,
    metric=autosklearn.metrics.f1_weighted, 
)

print(f"Fitting Classifier...")
classifier.fit(X_train, y_train)
print("Classifier fit complete.")

cv_results = pd.DataFrame(classifier.cv_results_)
cv_results.to_csv(os.path.join(OUTPUT_FOLDER, NOW+'_cv_results.csv'))
y_pred = classifier.predict(X_test)
results_df = pd.DataFrame({
    "y_true": y_test,
    'y_pred': y_pred
}).set_index(X_test.index)

acc = accuracy_score(y_test, y_pred)
print(f'Overall accuracy of {acc}')

fname = os.path.join(OUTPUT_FOLDER, NOW+'_autosklearn_results.csv')
results_df.to_csv(fname)

pickle.dump(classifier, open(os.path.join(OUTPUT_FOLDER, "classifier.P"), 'wb'))

print(
    f"Saved results to {fname} and model to {os.path.join(OUTPUT_FOLDER, 'classifier.P')}"
)






