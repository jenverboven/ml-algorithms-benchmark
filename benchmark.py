import streamlit as st
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import category_encoders as ce

st.title("Machine learning algorithms benchmark")

# Load dataset
tictactoe_df = pd.read_csv("tic-tac-toe.csv", sep=',')

st.write("Tic Tac Toe endgame dataframe specs:")
st.write(tictactoe_df.describe())

# Split the data
feature_cols = ['top-left-square', 'top-middle-square', 'top-right-square', 'middle-left-square', 'middle-middle-square', 'middle-right-square', 'bottom-left-square', 'bottom-middle-square', 'bottom-right-square']
target_variable = ['class']

x = tictactoe_df[feature_cols]
y = tictactoe_df[target_variable]

# Encode the data
ce_oh = ce.OneHotEncoder(cols=feature_cols)
x_cat = ce_oh.fit_transform(x)

# Split into training and test set
x_train, x_test, y_train, y_test = train_test_split(x_cat, y, test_size=0.2, random_state=42)

# Decision Tree button
if st.button("Decision Tree"):
    classifier = DecisionTreeClassifier(criterion="entropy")
    classifier = classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)
    accuracy_score_decision_tree = accuracy_score(y_test, prediction)
    st.write("Accuracy of Decision Tree: ", accuracy_score_decision_tree)

# Logistic Regression button
if st.button("Logistic Regression"):
    classifier = LogisticRegression()
    classifier = classifier.fit(x_train, y_train.values.ravel())
    prediction = classifier.predict(x_test)
    accuracyscore = accuracy_score(y_test, prediction)
    st.write("Accuracy of Logistic Regression: ", accuracyscore)

# SVC button
if st.button("SVC"):
    classifier = SVC()
    classifier = classifier.fit(x_train, y_train.values.ravel())
    prediction = classifier.predict(x_test)
    accuracyscore = accuracy_score(y_test, prediction)
    st.write("Accuracy of SVC: ", accuracyscore)
