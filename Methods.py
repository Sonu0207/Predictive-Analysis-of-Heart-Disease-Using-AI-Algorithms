from DecisionTree import *
from RandomForest import *
from LogisticRegression import *
from SVM import *
from NaiveBayes import *

if __name__ == '__main__':
    # read data file
    t_df = pd.read_csv("heartDiseaseTrainingData.csv")
    # drop columns with missing (not applicable) values
    t_df = t_df.dropna()
    # all data points to be used in analysis except true values
    X = t_df.drop(columns=['cardio'])
    # true values for which subjects have heart disease (1) and which don't (0)
    y = t_df['cardio']

    df = t_df

    # Count-plot for Age
    # rcParams['figure.figsize'] = 11, 8
    # sns.countplot(x='age_years', hue='cardio', data = df, palette="Set2")
    # # plt.show(block=True)
    #
    # # Countplot for Cholesterol, Glucose, Smoke, Alcohol, Active
    # df_long = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol','gluc', 'smoke', 'alco', 'active'])
    # sns.catplot(x="variable", hue="value", col="cardio",
    #                 data=df_long, kind="count")
    # plt.show(block=True)

    # Count gender
    # gender_count = df['gender'].value_counts()
    # # Create a pie chart for gender count
    # plt.figure(figsize=(8, 6))
    # plt.pie(gender_count, labels=gender_count.index, autopct='%1.1f%%', colors=['lightcoral', 'lightskyblue'])
    # plt.title('Distribution of Genders')
    # # Add a legend to specify that 1 is Female and 2 is Male
    # plt.legend(labels=['Female', 'Male'], loc='upper right')
    # plt.show(block=True)

    # Count gender alcohol
    # df['gender_mapped'] = df['gender'].map({1: 'Female', 2: 'Male'})
    # # Count gender alcohol
    # alcohol_count_by_gender = df.groupby('gender_mapped')['alco'].sum()
    # # Create a bar plot
    # plt.figure(figsize=(8, 6))
    # sns.barplot(x=alcohol_count_by_gender.index, y=alcohol_count_by_gender.values, palette='viridis')
    # plt.title('Alcohol Consumption by Gender')
    # plt.xlabel('Gender')
    # plt.ylabel('Total Alcohol Consumption')
    # plt.show(block=True)

    # Crosstab
    # cross_tab = pd.crosstab(df['cardio'], df['gender'], normalize=True)
    # # Create a heatmap using seaborn
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cross_tab, annot=True, fmt=".2%", cmap="YlGnBu", cbar_kws={'label': 'Percentage'})
    # plt.title('Cardio vs Gender Cross-Tabulation')
    # plt.xlabel('Gender')
    # plt.ylabel('Cardio')
    # plt.show(block=True)

    # Count the occurrences of each class
    class_counts = df['cardio'].value_counts()
    # Plot the count of 0 and 1 values with annotations
    # plt.figure(figsize=(8, 6))
    # ax = sns.countplot(x='cardio', data=df, palette='viridis')
    # # Annotate each bar with its count (integer, no decimal)
    # for p in ax.patches:
    #     ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
    #                 ha='center', va='center', fontsize=12, color='black', xytext=(0, 10),
    #                 textcoords='offset points')
    # plt.title('Count of 0 and 1 values in Cardio Column')
    # plt.xlabel('Cardio')
    # plt.ylabel('Count')
    # plt.show(block=True)

    # Mean height by gender
    df.groupby('gender')['height'].mean()

    # Remove outliers
    df.drop(df[(df['height'] > df['height'].quantile(0.975)) | (df['height'] < df['height'].quantile(0.025))].index,
            inplace=True)
    df.drop(df[(df['weight'] > df['weight'].quantile(0.975)) | (df['weight'] < df['weight'].quantile(0.025))].index,
            inplace=True)

    df.drop(df[(df['ap_hi'] > df['ap_hi'].quantile(0.975)) | (df['ap_hi'] < df['ap_hi'].quantile(0.025))].index,
            inplace=True)
    df.drop(df[(df['ap_lo'] > df['ap_lo'].quantile(0.975)) | (df['ap_lo'] < df['ap_lo'].quantile(0.025))].index,
            inplace=True)

    # blood_pressure = df.loc[:,['ap_lo','ap_hi']]
    # sns.boxplot(x = 'variable',y = 'value',data = blood_pressure.melt())
    # plt.show(block=True)
    # print("Diastilic pressure is higher than systolic one in {0} cases".format(df[df['ap_lo']> df['ap_hi']].shape[0]))

    # Separate majority and minority classes
    df_majority = df[df['cardio'] == 0]
    df_minority = df[df['cardio'] == 1]

    # Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=len(df_majority),  # to match majority class
                                     random_state=42)  # reproducible results

    # Combine majority class with upsampled minority class
    df_balanced = pd.concat([df_majority, df_minority_upsampled])

    # Shuffle the DataFrame to randomize the order of samples
    df = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Count the occurrences of each class
    # class_counts = df['cardio'].value_counts()
    # Plot the count of 0 and 1 values with annotations
    # plt.figure(figsize=(8, 6))
    # ax = sns.countplot(x='cardio', data=df, palette='viridis')
    # # Annotate each bar with its count (integer, no decimal)
    # for p in ax.patches:
    #     ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
    #                 ha='center', va='center', fontsize=12, color='black', xytext=(0, 10),
    #                 textcoords='offset points')
    # plt.title('Count of 0 and 1 values in Cardio Column')
    # plt.xlabel('Cardio')
    # plt.ylabel('Count')
    # plt.show(block=True)

    # df = df.drop(columns=['age_years', 'gender_mapped'])

    # Label Encoding
    le = LabelEncoder()
    df['bp_category'] = le.fit_transform(df['bp_category'])

    # Move 'cardio' column to the last position
    df = df[[col for col in df if col != 'cardio'] + ['cardio']]

    csv_path = r'C:\Users\julie\PycharmProjects\teamProject_Demo\Processed_heartDiseaseData.csv'

    # Export the DataFrame to a CSV file
    df.to_csv(csv_path, index=False)

    # read data file
    t_df = pd.read_csv("Processed_heartDiseaseData.csv")
    # drop columns with missing (not applicable) values
    t_df = t_df.dropna()
    # all data points to be used in analysis except true values
    X = t_df.drop(columns=['cardio'])
    # true values for which subjects have heart disease (1) and which don't (0)
    y = t_df['cardio']




    print("\nNaive Bayes")
    nb = basic(t_df)
    print("\nLogistic Regression")
    lr = logisticRegression(X, y)
    print("\nSVM")
    svm = svm(X, y)
    print("\nDecision Tree")
    dt = decisionTree(t_df)
    print("\nRandom Forest")
    rf = randomForest(t_df)

    # index = 0
    # accuracy = [nb[index], lr[index], svm[index], dt[index], rf[index]]
    # index = index + 1
    #
    # precision = [nb[index], lr[index], svm[index], dt[index], rf[index]]
    # index = index + 1
    #
    # recall = [nb[index], lr[index], svm[index], dt[index], rf[index]]
    # index = index + 1
    #
    # f1 = [nb[index], lr[index], svm[index], dt[index], rf[index]]
    # index = index + 1
    #
    # specificity = [nb[index], lr[index], svm[index], dt[index], rf[index]]
    # index = index + 1
    #
    # mcc = [nb[index], lr[index], svm[index], dt[index], rf[index]]
    # index = index + 1
    #
    # # print(accuracy)
    # # print(precision)
    # # print(recall)
    # # print(specificity)
    # # print(f1)
    # # print(mcc)
    #
    # keys = ['Naive Bayes', 'Logistic Regression', 'SVM', 'Decision Tree', 'Random Forest']
    # index = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score', 'MCC']
    # data = {'Naive Bayes': nb, 'Logistic Regression': lr, 'SVM': svm, 'Decision Tree': dt, 'Random Forest': rf}
    # dataframe = pd.DataFrame(data, columns=keys, index=index)
    # dataframe.plot.barh()
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.subplots_adjust(right=0.65)
    # plt.show()