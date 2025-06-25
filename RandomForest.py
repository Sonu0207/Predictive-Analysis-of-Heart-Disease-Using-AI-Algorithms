import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from matplotlib import rcParams

def randomForest(df):
    # Splitting the dataset
    X, X_test = train_test_split(df, test_size=.22, random_state=42)
    # print(X.shape, X_test.shape)

    # MODEL & PERFORMANCE METRICS
    # Metrics dictionary
    accuracy = dict()
    precision = dict()
    recall = dict()
    f1 = dict()
    specificity = dict()
    mcc = dict()

    # Random Forest
    sample = X
    y_sample = sample["cardio"]
    X_sample = sample.drop("cardio", axis=1)

    X_train, X_validate, y_train, y_validate = train_test_split(X_sample, y_sample, random_state=42)
    # print(X_train.shape, y_train.shape)
    # print(X_validate.shape, y_validate.shape)

    rfc = RandomForestClassifier(n_jobs=-1, random_state=42)
    parameters = [{"n_estimators": [50, 100, 150], "max_depth": [5, 10, 15, 30]}]
    # Define the number of folds (e.g., 10-fold cross-validation)
    n_folds = 10
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    # Create a GridSearchCV object with the specified number of folds
    grid = GridSearchCV(rfc, parameters, cv=kf, verbose=0, n_jobs=-1)
    grid.fit(X_train, y_train)

    # print("Best parameters scores:")
    # print(grid.best_params_)
    # print("Train score:", grid.score(X_train, y_train))
    # print("Validation score:", grid.score(X_validate, y_validate))

    # print("Default scores:")
    rfc.fit(X_train, y_train)
    # print("Train score:", rfc.score(X_train, y_train))
    # print("Validation score:", rfc.score(X_validate, y_validate))

    pd.DataFrame(grid.cv_results_).sort_values(by="rank_test_score")

    y_pred = rfc.predict(X_validate)

    accuracy["Random Forest"] = accuracy_score(y_validate, y_pred)
    f1["Random Forest"] = f1_score(y_validate, y_pred, average="macro")
    precision["Random Forest"] = precision_score(y_validate, y_pred, average="macro")
    recall["Random Forest"] = recall_score(y_validate, y_pred, average="macro")
    mcc["Random Forest"] = matthews_corrcoef(y_validate, y_pred)
    # print(classification_report(y_train, rfc.predict(X_train)))
    # print(classification_report(y_validate, y_pred))

    y_pred = rfc.predict(X_validate)
    confmat = confusion_matrix(y_true=y_validate, y_pred=y_pred)
    specificity["Random Forest"] = confmat[1, 1] / (confmat[1, 0] + confmat[1, 1])

    index = ["Actual No-Risk", "Actual Risk"]
    columns = ["Predicted No-Risk", "Predicted Risk"]
    conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
    print(conf_matrix)
    print(f"Accuracy: {accuracy['Random Forest'] * 100:.2f}%")
    print(f"Precision: {precision['Random Forest'] * 100:.2f}%")
    print(f"Recall: {recall['Random Forest'] * 100:.2f}%")
    print(f"F1-Score: {f1['Random Forest'] * 100:.2f}%")
    print(f"Specificity: {specificity['Random Forest'] * 100:.2f}%")
    print(f"MCC: {mcc['Random Forest']:.2f}")

    # plt.figure(figsize=(8, 5))
    # sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
    # plt.title("Confusion Matrix - Random Forest")
    # plt.show(block=True)

    return [accuracy["Random Forest"],
    precision["Random Forest"],
    recall["Random Forest"],
    f1["Random Forest"],
    specificity["Random Forest"],
    mcc["Random Forest"]]

def randomForestDemo(df, X_predict):
    y = df["cardio"]
    X = df.drop("cardio", axis=1)
    rfc = RandomForestClassifier(n_jobs=-1, random_state=42)
    parameters = [{"n_estimators": [50, 100, 150], "max_depth": [5, 10, 15, 30]}]
    n_folds = 10
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    grid = GridSearchCV(rfc, parameters, cv=kf, verbose=0, n_jobs=-1)
    grid.fit(X, y)
    rfc.fit(X, y)
    pd.DataFrame(grid.cv_results_).sort_values(by="rank_test_score")
    y_pred = rfc.predict(X_predict)
    return y_pred

if __name__ == "__main__":
    # read data file
    t_df = pd.read_csv("heartDiseaseTrainingData.csv")
    # drop columns with missing (not applicable) values
    t_df = t_df.dropna()
    print(randomForest(t_df))