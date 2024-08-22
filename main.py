import tkinter as tk
from tkinter import ttk
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore  # Add this import statement

# Function to load data and preprocess
def preprocess_data():
    global X_train, X_test, y_train, y_test
    data = pd.read_csv("sensor.csv")
    numerical_data = data.select_dtypes(include=['float64'])
    z_scores = numerical_data.apply(zscore)
    threshold = 3
    outliers = data[(z_scores.abs() > threshold).any(axis=1)]
    clean_data = data.drop(outliers.index)
    clean_data[numerical_data.columns] = clean_data[numerical_data.columns].apply(pd.to_numeric, errors='coerce')
    smoothed_data = clean_data.select_dtypes(include=['float64']).rolling(window=3, min_periods=1).mean()
    smoothed_data = pd.concat([clean_data.select_dtypes(exclude=['float64']), smoothed_data], axis=1)
    label_encoder = LabelEncoder()
    smoothed_data["machine_status"] = label_encoder.fit_transform(smoothed_data["machine_status"])
    smoothed_data['timestamp'] = pd.to_datetime(smoothed_data['timestamp'], format='%d-%m-%Y %H:%M')
    sensor_data = smoothed_data.drop(columns=["timestamp"])
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(sensor_data)
    standardized_df = pd.DataFrame(standardized_data, columns=sensor_data.columns)
    X = standardized_df.drop(columns=["machine_status"])
    Y = smoothed_data["machine_status"]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    return X_train_imputed, X_test_imputed, y_train, y_test

# Function to train and evaluate models
def train_and_evaluate_model(algorithm):
    if algorithm == "Decision Tree":
        model = DecisionTreeRegressor()
    elif algorithm == "Random Forest":
        model = RandomForestClassifier()
    elif algorithm == "SVM":
        model = SVC()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    if algorithm == "Decision Tree":
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return f"Mean Squared Error (MSE): {mse:.4f}\nMean Absolute Error (MAE): {mae:.4f}\nR^2 Score: {r2:.4f}"
    else:
        return f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1 Score: {f1:.4f}"

# Function to handle button click event
def on_click():
    selected_algorithm = combo_algorithm.get()
    result_text.set(train_and_evaluate_model(selected_algorithm))

# GUI
root = tk.Tk()
root.title("Machine Learning Model Performance")
root.geometry("400x300")

frame = tk.Frame(root)
frame.pack(pady=20)

# Load data
X_train, X_test, y_train, y_test = preprocess_data()

# Algorithm selection
label_algorithm = tk.Label(frame, text="Select Algorithm:")
label_algorithm.grid(row=0, column=0, padx=10, pady=5)

algorithms = ["Decision Tree", "Random Forest", "SVM"]
combo_algorithm = ttk.Combobox(frame, values=algorithms)
combo_algorithm.current(0)
combo_algorithm.grid(row=0, column=1, padx=10, pady=5)

# Button to trigger model training and evaluation
button = tk.Button(frame, text="Evaluate", command=on_click)
button.grid(row=1, column=0, columnspan=2, pady=10)

# Display result
result_text = tk.StringVar()
label_result = tk.Label(frame, textvariable=result_text, justify="left")
label_result.grid(row=2, column=0, columnspan=2, padx=10, pady=5)

root.mainloop()
