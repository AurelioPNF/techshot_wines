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

#Importing Data from Data_Manipulation
from pathlib import Path
import sys

class ModelCreator():

    def __init__(self, img_size=5) -> None:
        sys.path.append(str(Path(__file__).resolve().parent))
        from A_Data_Manipulation import Dados

        data = Dados()
        self.data = data
        self.img_size = img_size
        self.classes = ['Bad Wine', 'Good Wine']
        self.fit_model, self.X_train, self.X_test, self.y_train, self.y_test = self.trained_model(self.data.valid_data)

    def trained_model(self,df):
        X = df.drop(columns=['quality'])
        y = df['quality']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=111)

        fit_model = ExtraTreesClassifier(n_estimators=100, random_state=111,n_jobs=-1)
        fit_model.fit(X_train, y_train)
        return fit_model, X_train, X_test, y_train, y_test

    def score(self):
        return f'## Score: {(self.fit_model.score(self.X_test,self.y_test)*100):.2f}%'

    def con_matrix(self):
        fig = plt.figure(figsize=(self.img_size, self.img_size))
        confusion_matrix(
            self.fit_model,
            self.X_train, self.y_train, self.X_test, self.y_test,
            percent=True, classes=self.classes
        )
        plt.tight_layout()
        return fig

    def class_pred_plot(self):
        fig = plt.figure(figsize=(self.img_size, self.img_size))
        class_prediction_error(
            self.fit_model,
            self.X_train, self.y_train, self.X_test, self.y_test,
            classes=self.classes
        )
        plt.tight_layout()
        return fig

    def roc_plot(self):
        fig = plt.figure(figsize=(self.img_size, self.img_size))
        roc_auc(
            self.fit_model,
            self.X_train, self.y_train, self.X_test, self.y_test,
            classes=self.classes
        )
        plt.tight_layout()
        return fig

    def classification_report_plot(self):
        fig = plt.figure(figsize=(self.img_size, self.img_size))
        classification_report(
            self.fit_model,
            self.X_train, self.y_train, self.X_test, self.y_test,
            classes=self.classes
        )
        plt.tight_layout()
        return fig

    def precision_recall_plot(self):
        fig = plt.figure(figsize=(self.img_size, self.img_size))
        precision_recall_curve(
            self.fit_model,
            self.X_train, self.y_train, self.X_test, self.y_test,
            classes=self.classes
        )
        plt.tight_layout()
        return fig


model = ModelCreator(5)


st.write("## First, let's use pycaret to find the best model for our data:")
image = Image.open(f'{model.data.path_to}/best_model.png')
st.image(image, caption='Models tested by Pycaret')

st.write('#### As we can see, ExtraTreesClassifier is the best model')
st.write("### Let's use that to our advantage and create a model using Sklearn")

st.write("## Creating the model and analyzing its metrics")

# X = model.data.valid_data.drop(columns=['quality'])
# y = model.data.valid_data['quality']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=111)

# clf = ExtraTreesClassifier(n_estimators=100, random_state=111,n_jobs=-1)
# clf.fit(X_train, y_train)


st.write(model.score())
st.write("The score however isn't enough to evaluate the model, so let's see a few important metrics")
st.write('## Important Metrics')

st.write("### Confusion Matrix")
st.write(model.con_matrix())

st.write('### Class Prediction Error')
st.write(model.class_pred_plot())

st.write('### Roc Curve')
st.write(model.roc_plot())

st.write('### Classification Report')
st.write(model.classification_report_plot())

st.write('### Precision Recall Curve')
st.write(model.precision_recall_plot())

st.write('#### Looking at all the metrics, we can say that the model is very good at predicting if a wine is good, although not so much if a wine is bad.')