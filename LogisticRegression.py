import statsmodels.api as sm
from sklearn.metrics import *
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns


def logisticRegression(X, y):
    # use 80:20 ratio for training to testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    lr = LogisticRegression(random_state=42)
    lr_parameters = [{"penalty": ['l2'],
                      "C": [0.01, 0.1, 1, 10, 100],
                      "solver": ['liblinear', 'newton-cg', 'newton-cholesky']}]
    # Define the number of folds (e.g., 10-fold cross-validation)
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    # Create a GridSearchCV object with the specified number of folds
    lr_grid = GridSearchCV(lr, lr_parameters, cv=kf, verbose=0, n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    y_pred = lr_grid.predict(X_test)
    # convert all predictions to binary scale
    bin_predictions = [1 if x > 0.5 else 0 for x in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_test, bin_predictions).ravel()
    confmat = confusion_matrix(y_test, bin_predictions)
    index = ["Actual No-Risk", "Actual Risk"]
    columns = ["Predicted No-Risk", "Predicted Risk"]
    conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
    accuracy = accuracy_score(y_test, bin_predictions)
    precision = precision_score(y_test, bin_predictions)
    recall = recall_score(y_test, bin_predictions)
    specificity = (tn / (tn + fp))
    f1 = f1_score(y_test, bin_predictions)
    mcc = matthews_corrcoef(y_test, bin_predictions)
    print(conf_matrix)
    print('Accuracy: %f' % accuracy)
    print('Precision: %f' % precision)
    print('Recall: %f' % recall)
    print('Specificity: %f' % specificity)
    print('F1-Score: %f' % f1)
    print('Matthew Correlation Coefficient: %f' % mcc)

    # thresholds minimize the difference between false positive rate (fpr)
    # and true positive rate (tpr)
    # fpr, tpr, thresholds = roc_curve(y_test, predictions)
    # level to which classifier correctly identifies fpr and tpr with values
    # ranging from 0 (less accurate) to 1 (more accurate)
    # roc_auc = roc_auc_score(y_test, predictions)
    # plt.plot(fpr, tpr, label='ROC Curve (area = %0.3f)' % roc_auc)
    # plt.title('ROC Curve (area = %0.3f)' % roc_auc)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.show()

    # index = ["Actual No-Risk", "Actual Risk"]
    # columns = ["Predicted No-Risk", "Predicted Risk"]
    # cm = confusion_matrix(y_test, bin_predictions)
    # conf_matrix = pd.DataFrame(data=cm, columns=columns, index=index)
    # plt.figure(figsize=(8, 5))
    # sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
    # plt.title("Confusion Matrix - Logistic Regression")
    # plt.show(block=True)
    # plt.show()

    return [accuracy, precision, recall, f1, specificity, mcc]

def logisticRegressionDemo(X, y, X_predict):
    # build model and train it on the training data
    logmodel = sm.Logit(y, sm.add_constant(X)).fit(disp=False)
    # produce probabilities (0:1 inclusive) that subjects have heart disease
    predictions = logmodel.predict(sm.add_constant(X_predict))
    print(predictions)
    # convert all predictions to binary scale
    bin_predictions = [1 if x > 0.5 else 0 for x in predictions]
    return bin_predictions

if __name__ == "__main__":
    # read data file
    t_df = pd.read_csv("heartDiseaseTrainingData.csv")
    # drop columns with missing (not applicable) values
    t_df = t_df.dropna()
    # all data points to be used in analysis except true values
    X = t_df.drop(columns=['cardio'])
    # true values for which subjects have heart disease (1) and which don't (0)
    y = t_df['cardio']
    print(logisticRegression(X, y))