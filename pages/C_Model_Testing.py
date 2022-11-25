import streamlit as st
import pandas as pd


#Importing Data from Data_Manipulation
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from B_Model_Creation import ModelCreator


model = ModelCreator(5)


st.markdown("# Model Testing")
st.markdown('## Insert the parameters below to generate a prediction')

data = [x for x in range(len(model.X_test.columns))]
prediction_data = []
for index, x in enumerate(model.X_test.columns):
    data[index] = st.slider(f'{x}', model.X_test[x].min(), model.X_test[x].max(),float(model.X_test[x].iloc[0]),step=0.01)
    prediction_data.append(data[index])
    

prediction_data = pd.DataFrame(prediction_data)

st.markdown('## Prediction values')
st.write(prediction_data.T)


prediction = model.fit_model.predict(prediction_data.T)[0]

if prediction == 1:
    quality = 'good'
else:
    quality = 'bad'

st.markdown(f"## According to the prediction, this is a {quality} wine!")