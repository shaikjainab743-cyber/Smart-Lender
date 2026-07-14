from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("model/loan_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    gender = 1 if request.form["Gender"] == "Male" else 0
    married = 1 if request.form["Married"] == "Yes" else 0

    dependents = request.form["Dependents"]
    dep = {"0": 0, "1": 1, "2": 2, "3+": 3}
    dependents = dep.get(dependents, 0)

    education = 1 if request.form["Education"] == "Graduate" else 0

    self_employed = 1 if request.form["Self_Employed"] == "Yes" else 0

    applicant_income = float(request.form["ApplicantIncome"])
    coapplicant_income = float(request.form["CoapplicantIncome"])
    loan_amount = float(request.form["LoanAmount"])
    loan_term = float(request.form["Loan_Amount_Term"])
    credit_history = float(request.form["Credit_History"])

    area = request.form["Property_Area"]
    area_map = {
        "Rural": 0,
        "Semiurban": 1,
        "Urban": 2
    }

    property_area = area_map[area]

    # Loan_ID placeholder
    loan_id = 0

    data = pd.DataFrame([[
        loan_id,
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_area
    ]])

    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "✅ Congratulations! Loan Approved"
    else:
        result = "❌ Sorry! Loan Rejected"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)
