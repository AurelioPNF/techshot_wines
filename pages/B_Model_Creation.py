import streamlit as st

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from yellowbrick.classifier import confusion_matrix
from yellowbrick.classifier import classification_report
from yellowbrick.classifier.rocauc import roc_auc
from yellowbrick.classifier import precision_recall_curve
from yellowbrick.classifier import class_prediction_error

from PIL import Image

def con_matrix(model):
    fig = plt.figure(figsize=(5, 5))
    confusion_matrix(
        model,
        X_train, y_train, X_test, y_test,
        percent=True, classes=['Bad Wine', 'Good Wine']
    )
    plt.tight_layout()
    return fig

def class_pred_plot(model):
    fig = plt.figure(figsize=(5, 5))
    class_prediction_error(
        model,
        X_train, y_train, X_test, y_test,
        classes=['Bad Wine', 'Good Wine']
    )
    plt.tight_layout()
    return fig

def roc_plot(model):
    fig = plt.figure(figsize=(5, 5))
    roc_auc(
        model,
        X_train, y_train, X_test, y_test,
        classes=['Bad Wine', 'Good Wine']
    )
    plt.tight_layout()
    return fig

def classification_report_plot(model):
    fig = plt.figure(figsize=(5, 5))
    classification_report(
        model,
        X_train, y_train, X_test, y_test,
        classes=['Bad Wine', 'Good Wine']
    )
    plt.tight_layout()
    return fig

def precision_recall_plot(model):
    fig = plt.figure(figsize=(5, 5))
    precision_recall_curve(
        model,
        X_train, y_train, X_test, y_test,
        classes=['Bad Wine', 'Good Wine']
    )
    plt.tight_layout()
    return fig

#Importing Data from Data_Manipulation
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from A_Data_Manipulation import Dados

data = Dados()

st.write("## First, let's use pycaret to find the best model for our data:")
image = Image.open(f'{data.path_to}/best_model.png')
st.image(image, caption='Models tested by Pycaret')

st.write('#### As we can see, ExtraTreesClassifier is the best model')
st.write("### Let's use that to our advantage and create a model using Sklearn")

st.write("## Creating the model and analyzing its metrics")

X = data.valid_data.drop(columns=['quality'])
y = data.valid_data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=111)

clf = ExtraTreesClassifier(n_estimators=100, random_state=111,n_jobs=-1)
clf.fit(X_train, y_train)


st.write(f'## Score: {(clf.score(X_test,y_test)*100):.2f}%')
st.write("The score however isn't enough to evaluate the model, so let's see a few important metrics")
st.write('## Important Metrics')

st.write("### Confusion Matrix")
st.write(con_matrix(clf))

st.write('### Class Prediction Error')
st.write(class_pred_plot(clf))

st.write('### Roc Curve')
st.write(roc_plot(clf))

st.write('### Classification Report')
st.write(classification_report_plot(clf))

st.write('### Precision Recall Curve')
st.write(precision_recall_plot(clf))

st.write('#### Looking at all the metrics, we can say that the model is very good at predicting if a wine is good, although not so much if a wine is bad.')