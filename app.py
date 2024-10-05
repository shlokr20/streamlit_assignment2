import streamlit as st
import streamlit_shadcn_ui as ui
import pickle
import os
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Titanic Dataset
def load_titanic_data():
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    titanic_data = pd.read_csv(url)
    titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})  # Convert categorical to numeric
    titanic_data = titanic_data.dropna(subset=['Age', 'Fare'])  # Drop rows with missing values
    return titanic_data
 
# 1. Train Linear Regression Model to predict Fare
def train_linear_regression(titanic_data):
    X = titanic_data[['Pclass', 'Age', 'Sex']]
    y = titanic_data['Fare']
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    model = LinearRegression()
    model.fit(X_train, y_train)
 
    # Save the model in the same directory as app.py
    with open('linear_regression_model.pkl', 'wb') as f:
        pickle.dump(model, f)
 
    print("Linear Regression model saved.")
 
# 2. Train Logistic Regression Model to predict Survival
def train_logistic_regression(titanic_data):
    X = titanic_data[['Pclass', 'Age', 'Fare', 'Sex']]
    y = titanic_data['Survived']
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
 
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")
 
    # Save the model in the same directory as app.py
    with open('logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(model, f)
 
    print("Logistic Regression model saved.")
 
# 3. Train Naive Bayes Model to predict Survival
def train_naive_bayes(titanic_data):
    X = titanic_data[['Pclass', 'Age', 'Fare', 'Sex']]
    y = titanic_data['Survived']
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    model = GaussianNB()
    model.fit(X_train, y_train)
 
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Naive Bayes Accuracy: {accuracy:.4f}")
 
    # Save the model in the same directory as app.py
    with open('naive_bayes_model.pkl', 'wb') as f:
        pickle.dump(model, f)
 
    print("Naive Bayes model saved.")
 
# 4. Train Decision Tree Model to predict Survival
def train_decision_tree(titanic_data):
    X = titanic_data[['Pclass', 'Age', 'Fare', 'Sex']]
    y = titanic_data['Survived']
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
 
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision Tree Accuracy: {accuracy:.4f}")
 
    # Save the model in the same directory as app.py
    with open('decision_tree_model.pkl', 'wb') as f:
        pickle.dump(model, f)
 
    print("Decision Tree model saved.")
 
# 5. Apriori Algorithm for Recommendation System (based on Survival, Pclass, and Sex)
def train_apriori(titanic_data):
    # Use relevant binary features (1: present, 0: not present) for Apriori
    titanic_data['Survived'] = titanic_data['Survived'].apply(lambda x: 1 if x == 1 else 0)
    titanic_data['Pclass_1'] = titanic_data['Pclass'].apply(lambda x: 1 if x == 1 else 0)
    titanic_data['Pclass_2'] = titanic_data['Pclass'].apply(lambda x: 1 if x == 2 else 0)
    titanic_data['Pclass_3'] = titanic_data['Pclass'].apply(lambda x: 1 if x == 3 else 0)
   
    data_for_apriori = titanic_data[['Survived', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex']]
   
    # Apply the Apriori algorithm
    frequent_itemsets = apriori(data_for_apriori, min_support=0.2, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
 
    print(f"Apriori Rules: \n{rules.head()}")
 
    # Save the results in the same directory as app.py
    with open('apriori_model.pkl', 'wb') as f:
        pickle.dump(frequent_itemsets, f)
 
    print("Apriori model saved.")
 
# Load a model from the same directory
def load_model(model_name):
    with open(model_name, 'rb') as f:
        return pickle.load(f)
 
# Inferencing Functions
def predict_with_linear_regression(model, inputs):
    return model.predict([inputs])[0]
 
def predict_with_logistic_regression(model, inputs):
    return model.predict([inputs])[0]
 
def predict_with_naive_bayes(model, inputs):
    return model.predict([inputs])[0]
 
def predict_with_decision_tree(model, inputs):
    return model.predict([inputs])[0]
 
def load_apriori_rules():
    with open('apriori_model.pkl', 'rb') as f:
        return pickle.load(f)
