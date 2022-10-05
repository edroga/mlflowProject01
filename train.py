import warnings

import numpy as np
import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

import mlflow.sklearn


def acquire_training_data():
    df_output = pd.read_csv('https://raw.githubusercontent.com/edroga/Datasets_for_projects/main/Heart_Attack.csv', na_values='?')

    return df_output


def cleaning(df):
    df_clean = (df.drop(columns=['slope', 'ca', 'thal'])
                  .rename(columns={'num       ': 'target'})
                  .dropna())

    return df_clean


def transform_df(df):

    df_transformed = pd.get_dummies(df, columns=['cp', 'restecg'], drop_first=True)

    return df_transformed

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    with mlflow.start_run():

        training_data = acquire_training_data()
        cleaned_data = cleaning(training_data)
        transformed_data = transform_df(cleaned_data)

        X = transformed_data.drop(columns=['target'])
        Y = transformed_data['target']

        X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                            test_size=0.25,
                                                            random_state=4284,
                                                            stratify=Y)

        clf = RandomForestClassifier(bootstrap=True, criterion='gini', min_samples_split=2, min_weight_fraction_leaf=0.0, n_estimators=50, random_state=4284, verbose=0)

        clf.fit(X_train, y_train)

        predicted = clf.predict(X_test)

        mlflow.sklearn.log_model(clf, "model_random_forest")

        print(classification_report(y_test, predicted))

        mlflow.log_metric("precision_label_0", precision_score(y_test, predicted, pos_label=0))
        mlflow.log_metric("recall_label_0", recall_score(y_test, predicted, pos_label=0))
        mlflow.log_metric("f1score_label_0", f1_score(y_test, predicted, pos_label=0))
        mlflow.log_metric("precision_label_1", precision_score(y_test, predicted, pos_label=1))
        mlflow.log_metric("recall_label_1", recall_score(y_test, predicted, pos_label=1))
        mlflow.log_metric("f1score_label_1", f1_score(y_test, predicted, pos_label=1))













