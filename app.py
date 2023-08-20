from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("trained_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        # Get user input from the form
        age = int(request.form["age"])
        bmi = float(request.form["bmi"])
        dietary_habits = request.form["dietary_habits"]
        smoking = request.form["smoking"]
        medical_history = request.form["medical_history"]
        stress = request.form["stress"]
        medication = request.form["medication"]
        family_history = request.form["family_history"]
        sleep_patterns = request.form["sleep_patterns"]
        hydration = request.form["hydration"]

        # Create input data dictionary
        input_data = {
            "Age": age,
            "BMI": bmi,
            "Dietary Habits": dietary_habits,
            "Smoking": smoking,
            "Medical History": medical_history,
            "Stress": stress,
            "Medication": medication,
            "Family History": family_history,
            "Sleep Patterns": sleep_patterns,
            "Hydration": hydration,
        }

        # Convert input data into a DataFrame and encode using one-hot encoding
        encoded_input = pd.get_dummies(pd.DataFrame([input_data]))
        if "Stress_High" in encoded_input.columns:
            encoded_input["Stress_Low"] = False
            encoded_input["Stress_Medium"] = False
        elif "Stress_Low" in encoded_input.columns:
            encoded_input["Stress_High"] = False
            encoded_input["Stress_Medium"] = False
        else:
            encoded_input["Stress_High"] = False
            encoded_input["Stress_Low"] = False

        if "Smoking_No" in encoded_input.columns:
            encoded_input["Smoking_Yes"] = False
        else:
            encoded_input["Smoking_No"] = False

        if "Dietary Habits_Balanced" in encoded_input.columns:
            encoded_input["Dietary Habits_Spicy foods"] = False
        else:
            encoded_input["Dietary Habits_Balanced"] = False

        if "Family History_No" in encoded_input.columns:
            encoded_input["Family History_Yes"] = False
        else:
            encoded_input["Family History_No"] = False

        if "Sleep Patterns_Good" in encoded_input.columns:
            encoded_input["Sleep Patterns_Poor"] = False
        else:
            encoded_input["Sleep Patterns_Good"] = False

        if "Medical History_GERD" in encoded_input.columns:
            encoded_input["Medical History_Gastritis"] = False
            encoded_input["Medical History_Ulcer history"] = False
        elif "Medical History_Gastritis" in encoded_input.columns:
            encoded_input["Medical History_GERD"] = False
            encoded_input["Medical History_Ulcer history"] = False
        elif "Medical History_Ulcer history" in encoded_input.columns:
            encoded_input["Medical History_GERD"] = False
            encoded_input["Medical History_Gastritis"] = False

        # Ensure all columns are present
        missing_cols = set(model.feature_importances_) - set(encoded_input.columns)
        for col in missing_cols:
            encoded_input[col] = False

        # Reorder columns to match the desired order
        desired_order = [
            "Age", "BMI", "Dietary Habits_Balanced", "Dietary Habits_Spicy foods",
            "Smoking_No", "Smoking_Yes", "Medical History_GERD", "Medical History_Gastritis", "Medical History_Ulcer history", "Stress_High", "Stress_Low",
            "Stress_Medium", "Medication_NSAIDs", "Family History_No",
            "Family History_Yes", "Sleep Patterns_Good", "Sleep Patterns_Poor", "Hydration_Normal"
        ]
        encoded_input = encoded_input[desired_order]

        print(encoded_input.head())
        print(encoded_input.shape)

        # Make prediction using the model
        predicted_risk_level = model.predict(encoded_input)[0]

        # Convert prediction to appropriate format
        if predicted_risk_level == "Low":
            prediction = "Low Risk"
        elif predicted_risk_level == "Medium":
            prediction = "Medium Risk"
        elif predicted_risk_level == "High":
            prediction = "High Risk"
        else:
            prediction = "Loading..."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
