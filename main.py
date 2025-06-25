# from RandomForest import *
from LogisticRegression import *
from RandomForest import *
from flask import Flask, render_template, request

app = Flask(__name__)
@app.route("/")

def index():
    return render_template("index.html")

@app.route('/process', methods=['POST'])
def process():
    # read data file
    t_df = pd.read_csv("heartDiseaseTrainingData.csv")
    # drop columns with missing (not applicable) values
    t_df = t_df.dropna()
    # all data points to be used in analysis except true values
    X = t_df.drop(columns=['cardio'])
    # true values for which subjects have heart disease (1) and which don't (0)
    y = t_df['cardio']

    gender = int(request.form.get('gender'))
    age = int(request.form.get('age'))
    height = int(request.form.get('height'))
    weight = int(request.form.get('weight'))
    ap_hi = int(request.form.get('ap_hi'))
    ap_lo = int(request.form.get('ap_lo'))
    cholesterol = int(request.form.get('cholesterol'))
    gluc = int(request.form.get('gluc'))
    smoke = int(request.form.get('smoke'))
    alco = int(request.form.get('alco'))
    active = int(request.form.get('active'))
    bmi = float(request.form.get('bmi'))

    X_predict = [gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, age,  bmi, 0]
    print(X_predict)
    X_predict = np.column_stack(X_predict)
    # train the model, then predict outcome using user input data
    result = randomForestDemo(t_df, X_predict)
    print(result)
    prediction = ""
    if(result[0] <= 0.5):
        prediction = "Not at risk"
    else:
        prediction = "At risk"
    return str(prediction)

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug=True)