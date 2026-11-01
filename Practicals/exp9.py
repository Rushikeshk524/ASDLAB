#Question / Aim: Develop a machine learning classification model to predict whether an employee will Stay or Leave the organization based on Years of Experience, Monthly Salary, and Job Satisfaction Score.

#Theory: Machine learning is used to predict whether an employee will stay or leave an organization based on Years of Experience, Monthly Salary and Job Satisfaction Score. The dataset is divided into training and testing sets. A Decision Tree Classifier is trained using the training data. The model predicts employee attrition and its accuracy is calculated. MLflow is used to track model parameters, accuracy and trained models by changing the max_depth value.

#Result(Optional): The Employee Attrition Prediction model was successfully developed using a Decision Tree Classifier. Its accuracy was calculated, and two experiments with different max_depth values were recorded in MLflow.

#Conclusion(Optional): Thus, a Decision Tree Classifier was successfully implemented to predict employee attrition, and model performance was compared using MLflow.

# Step 0: check if pc has python and also download exp9.csv from repo
# Step 1(run this in terminal): pip install pandas scikit-learn mlflow 
# Step 2(run this in terminal): python exp9.py 
# Step 3(run this in terminal): mlflow ui

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


df = pd.read_csv('exp9.csv', skipinitialspace=True)

X = df[['Experience', 'Salary', 'Satisfaction']]
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

mlflow.set_experiment("Employee Attrition")

for depth in [2, 3]:

    with mlflow.start_run():

        model = DecisionTreeClassifier(max_depth=depth)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_param("max_depth", depth)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print("Max Depth:", depth)
        print("Accuracy:", accuracy)

print("Experiment completed!")
