from tpot import TPOTClassifier
from procan_connectome.config import DATA_PATH, RANDOM_STATE, LOGGER_LEVEL
import pandas as pd 
import numpy as np 
import os
from sklearn.model_selection import train_test_split
import datetime
from sklearn.metrics import accuracy_score
import pickle
import logging
from procan_connectome.utils.load_dataset import get_rf_dataset



NOW = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}"
OUTPUT_FOLDER = os.path.join(DATA_PATH, 'tpot_output',NOW+'_important_features_TPOT_output')
os.mkdir(OUTPUT_FOLDER)

log_file_name = NOW+"_TPOT"
log_file = os.path.join(OUTPUT_FOLDER, log_file_name + "_TPOT_LOGS")

logging.basicConfig(
    filename=os.path.join(OUTPUT_FOLDER, log_file_name + "_SCRIPT_LOGS"),
    filemode='a',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=LOGGER_LEVEL
)

# df = pd.read_csv(os.path.join(DATA_PATH, "combined_datasets.csv"))
# df = df.set_index('ID')
df = get_rf_dataset(threshold=0.001)
X, y = df.drop(columns='label'), df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y, 
    test_size=0.33,
    random_state=RANDOM_STATE
)

logging.info(f"Loaded X_train and X_test with respective shapes: {X_train.shape} and {X_test.shape}")

classifier = TPOTClassifier(
    n_jobs=-1, 
    scoring='f1_weighted',
    max_time_mins=1440,  # Set to 720 for 12 hours
    random_state=RANDOM_STATE, 
    verbosity=1, 
    log_file=log_file
)

logging.info(f"Fitting Classifier...")
classifier.fit(X_train, y_train)
logging.info("Classifier fit complete.")

y_pred = classifier.predict(X_test)
results_df = pd.DataFrame({
    "y_true": y_test,
    'y_pred': y_pred
}).set_index(X_test.index)

fname = os.path.join(DATA_PATH, OUTPUT_FOLDER, NOW+'_TPOT_results.csv')
results_df.to_csv(fname)
pickle.dump(classifier.fitted_pipeline_, open(os.path.join(DATA_PATH, OUTPUT_FOLDER, "classifier.P"), 'wb'))


logging.info(
    f"Saved results to {fname} and best pipeline to {os.path.join(DATA_PATH, OUTPUT_FOLDER, 'classifier.P')}"
)

acc = accuracy_score(y_test, y_pred)
logging.info(f'Overall accuracy of {acc}')

out_name = os.path.join(OUTPUT_FOLDER, 'exported_code.py')
classifier.export(out_name)
logging.info(f'Pipeline exported to {out_name}')




