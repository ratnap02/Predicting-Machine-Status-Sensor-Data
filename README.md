# link to download sensor dataset
> https://drive.google.com/file/d/1w7gamh6ch3NKNPC7wbQtJ1ecxxK_6zBj/view?usp=sharing
Predictive Maintenance System

This project involves developing a predictive maintenance system to forecast equipment conditions and minimize downtime using machine learning models. The goal is to predict when equipment might fail based on sensor data, enabling timely maintenance and reducing unplanned interruptions.

Key Components of the Project:

Data Preprocessing:
Data Cleaning: Loaded sensor data, identified and removed outliers using z-scores, and smoothed data to reduce noise.
Feature Engineering: Transformed categorical labels into numerical values using LabelEncoder and standardized the features using StandardScaler.
Data Splitting: Split data into training and test sets to evaluate model performance.

Model Training and Evaluation:
Algorithms Used:
Decision Tree Regressor: For regression tasks, evaluating metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² Score.

Random Forest Classifier: For classification tasks, evaluating metrics such as Accuracy, Precision, Recall, and F1 Score.

Support Vector Machine (SVM): Also used for classification, with similar evaluation metrics as Random Forest.

Performance Metrics: The system calculates and displays model performance metrics based on the selected algorithm.

User Interface:
Developed a graphical user interface (GUI) using Tkinter that allows users to select a machine learning algorithm, train the model, and view evaluation results interactively.



Detailed Actions Taken:
Preprocessing: Implemented data cleaning, smoothing, and transformation techniques to prepare the sensor data for model training.

Model Training: Trained multiple models (Decision Tree, Random Forest, SVM) and evaluated their performance using appropriate metrics.

GUI Development: Created a Tkinter-based application to facilitate model selection and performance evaluation, making the system user-friendly and accessible.


