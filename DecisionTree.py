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

def decisionTree(df):
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


    # Decision Tree
    sample = X
    y_sample = sample["cardio"]
    X_sample = sample.drop("cardio", axis=1)
    X_train, X_validate, y_train, y_validate = train_test_split(X_sample, y_sample, random_state=42)
    # print(X_train.shape, y_train.shape)
    # print(X_validate.shape, y_validate.shape)

    dtc = DecisionTreeClassifier(random_state=42)
    parameters = [{"criterion": ["gini", "entropy"], "max_depth": [5, 10, 15, 30]}]
    # Define the number of folds (e.g., 10-fold cross-validation)
    n_folds = 10
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    # Create a GridSearchCV object with the specified number of folds
    grid = GridSearchCV(dtc, parameters, cv=kf, verbose=0, n_jobs=-1)
    grid.fit(X_train, y_train)

    # print("Best parameters scores:")
    # print(grid.best_params_)
    # print("Train score:", grid.score(X_train, y_train))
    # print("Validation score:", grid.score(X_validate, y_validate))
    #
    # print("Base scores:")
    dtc.fit(X_train, y_train)
    # print("Train score:", dtc.score(X_train, y_train))
    # print("Validation score:", dtc.score(X_validate, y_validate))

    pd.DataFrame(grid.cv_results_).sort_values(by="rank_test_score")

    y_pred = dtc.predict(X_validate)

    accuracy["Decision Tree"] = accuracy_score(y_validate, y_pred)
    f1["Decision Tree"] = f1_score(y_validate, y_pred, average="macro")
    precision["Decision Tree"] = precision_score(y_validate, y_pred, average="macro")
    recall["Decision Tree"] = recall_score(y_validate, y_pred, average="macro")

    # print(classification_report(y_train, dtc.predict(X_train)))
    # print(classification_report(y_validate, y_pred))

    y_pred = dtc.predict(X_validate)
    confmat = confusion_matrix(y_true=y_validate, y_pred=y_pred)

    index = ["Actual No-Risk", "Actual Risk"]
    columns = ["Predicted No-Risk", "Predicted Risk"]
    conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
    print(conf_matrix)


    # Calculate Specificity
    specificity["Decision Tree"] = confmat[1, 1] / (confmat[1, 0] + confmat[1, 1])

    # Calculate MCC (Matthews Correlation Coefficient)
    mcc["Decision Tree"] = matthews_corrcoef(y_validate, y_pred)
    print(f"Accuracy: {accuracy['Decision Tree']*100:.2f}%")
    print(f"Precision: {precision['Decision Tree']*100:.2f}%")
    print(f"Recall: {recall['Decision Tree']*100:.2f}%")
    print(f"F1-Score: {f1['Decision Tree']*100:.2f}%")
    print(f"Specificity: {specificity['Decision Tree']*100:.2f}%")
    print(f"MCC: {mcc['Decision Tree']:.2f}")


    # plt.figure(figsize=(8, 5))
    # sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
    # plt.title("Confusion Matrix - Decision Tree")
    # plt.show(block=True)


    # importances = pd.DataFrame(np.zeros((X_train.shape[1], 1)), columns=["importance"], index=X_train.columns)
    #
    # importances.iloc[:,0] = dtc.feature_importances_
    #
    # importances = importances.sort_values(by="importance", ascending=False)

    # plt.figure(figsize=(15, 10))
    # sns.barplot(x="importance", y=importances.index, data=importances)
    # plt.title("Feature Importance - Decision Tree")
    # plt.show(block=True)

    return [accuracy["Decision Tree"],
            precision["Decision Tree"],
            recall["Decision Tree"],
            f1["Decision Tree"],
            specificity["Decision Tree"],
            mcc["Decision Tree"]]

if __name__ == "__main__":
    # read data file
    t_df = pd.read_csv("heartDiseaseTrainingData.csv")
    # drop columns with missing (not applicable) values
    t_df = t_df.dropna()
    print(decisionTree(t_df))