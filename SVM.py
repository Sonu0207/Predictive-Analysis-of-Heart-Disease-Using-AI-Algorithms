import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import *


def svm(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create an SVM model
    svm_model = SVC(kernel='linear')

    # # Define the number of folds (e.g., 10-fold cross-validation)
    # n_folds = 5
    # kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    # svm_parameters = [{'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
    #                    'gamma': ['scale', 'auto']}]
    # # Create a GridSearchCV object with the specified number of folds
    # svm_grid = GridSearchCV(svm_model, svm_parameters, cv=kf, verbose=0, n_jobs=-1)

    # Train the model
    svm_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = svm_model.predict(X_test)

    # Evaluate the model
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    confmat = confusion_matrix(y_test, y_pred)
    index = ["Actual No-Risk", "Actual Risk"]
    columns = ["Predicted No-Risk", "Predicted Risk"]
    conf_matrix = pd.DataFrame(data=confmat, columns=columns, index=index)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    specificity = (tn / (tn + fp))
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    print(conf_matrix)
    print('Accuracy: %f' % accuracy)
    print('Precision: %f' % precision)
    print('Recall: %f' % recall)
    print('Specificity: %f' % specificity)
    print('F1-Score: %f' % f1)
    print('Matthew Correlation Coefficient: %f' % mcc)

    # index = ["Actual No-Risk", "Actual Risk"]
    # columns = ["Predicted No-Risk", "Predicted Risk"]
    # cm = confusion_matrix(y_test, y_pred)
    # conf_matrix = pd.DataFrame(data=cm, columns=columns, index=index)
    # plt.figure(figsize=(8, 5))
    # sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
    # plt.title("Confusion Matrix - SVM")
    # plt.show(block=True)
    # plt.show()

    return [accuracy, precision, recall, f1, specificity, mcc]

if __name__ == "__main__":
    # read data file
    t_df = pd.read_csv("heartDiseaseTrainingData.csv")
    # drop columns with missing (not applicable) values
    t_df = t_df.dropna()
    # all data points to be used in analysis except true values
    X = t_df.drop(columns=['cardio'])
    # true values for which subjects have heart disease (1) and which don't (0)
    y = t_df['cardio']
    svm(X, y)