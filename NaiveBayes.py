import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV


def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    return df


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def evaluate_metrics(y_test, weighted_predictions):
    accuracy = accuracy_score(y_test, weighted_predictions)
    precision = precision_score(y_test, weighted_predictions, zero_division=0)
    recall = recall_score(y_test, weighted_predictions, zero_division=0)
    specificity = specificity_score(y_test, weighted_predictions)
    f1 = f1_score(y_test, weighted_predictions, zero_division=0)
    mcc = matthews_corrcoef(y_test, weighted_predictions)
    cm = confusion_matrix(y_test, weighted_predictions)
    return accuracy, precision, recall, specificity, f1, mcc, cm


def tune_hyperparameters(model, params, X_train, y_train):
    grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    # print(f'Best Parameters for {model.__class__.__name__}: {best_params}, Score: {best_score:.4f}')
    return grid_search.best_estimator_


def basic(df):
    continuous_columns = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi']
    for column in continuous_columns:
        df = remove_outliers(df, column)

    X = df.drop(columns=['cardio'])
    y = df['cardio']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)

    X_train_continuous = X_train[continuous_columns]
    X_train_categorical = X_train[['cholesterol', 'gluc']]
    X_train_binary = X_train[['gender', 'smoke', 'alco', 'active']]

    X_test_continuous = X_test[continuous_columns]
    X_test_categorical = X_test[['cholesterol', 'gluc']]
    X_test_binary = X_test[['gender', 'smoke', 'alco', 'active']]

    gnb = GaussianNB()

    params_bnb = {'alpha': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]}
    params_cnb = {'alpha': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]}

    bnb = tune_hyperparameters(BernoulliNB(), params_bnb, X_train_binary, y_train)
    cnb = tune_hyperparameters(CategoricalNB(), params_cnb, X_train_categorical, y_train)
    gnb.fit(X_train_continuous, y_train)

    probas_gnb = gnb.predict_proba(X_test_continuous)[:, 1]
    probas_cnb = cnb.predict_proba(X_test_categorical)[:, 1]
    probas_bnb = bnb.predict_proba(X_test_binary)[:, 1]

    weighted_probas = (probas_gnb * 0.2 + probas_cnb * 0.3 + probas_bnb * 0.5)
    weighted_predictions = (weighted_probas > 0.5).astype(int)

    accuracy, precision, recall, specificity, f1, mcc, cm = evaluate_metrics(y_test, weighted_predictions)

    confmat = cm
    index = ["Actual No-Risk", "Actual Risk"]
    columns = ["Predicted No-Risk", "Predicted Risk"]
    conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
    print(conf_matrix)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'Specificity: {specificity:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'MCC: {mcc:.4f}')

    return [accuracy, precision, recall, f1, specificity, mcc]

if __name__ == "__main__":
    t_df = pd.read_csv("heartDiseaseTrainingData.csv")
    basic(t_df)
