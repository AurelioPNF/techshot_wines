import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from pathlib import Path
import sys

# Importing BeautifulSoup class from the bs4 module
from bs4 import BeautifulSoup

# Importing the HTTP library
import requests as req

class Dados:

    def __init__(self) -> None:
        self.path_to = str(Path(__file__).resolve().parent.parent)
        self.df = pd.read_csv(f'{self.path_to}/archive/winequality-white.csv',sep=';')
        self.valid_data = self.df.copy()
        self.data_update()

    def basic_info(self,df):
        s = ""
        count = 1
        for c in df.columns:
            nan = df[c].isna().sum()
            shape = df.shape[0]

            
            s+= f"{count} - {c}\n"
            s+= f"    Type: {df[c].dtype}\n"
            s+= f"    No. of unique values {df[c].nunique()}\n"
            s+= f"    Sample of unique values {df[c].unique()[0:10]}\n"
            s+= f"    Null values {nan} ({np.round((nan/shape)*100, 2)} %)\n"
            s+= "\n"
            count += 1
        return s

    def quality_distribution(self,df):
        list_quality = sorted(df['quality'].unique())
        return_list = []
        for x in list_quality:
            return_list.append((f"Quality:{x} = {len(df[df['quality'] == x].index)/len(df.index)*100:.2f}%"))
        return return_list

    def plot_hist_quality(self,df):
        list_quality = sorted(df['quality'].unique())
        fig, ax = plt.subplots()
        ax = df['quality'].hist(bins=len(list_quality))
        ax.set_xlabel('Quality')
        ax.set_ylabel('Quantity')
        ax.set_xticks(list_quality)
        return fig

    def plot_hist_all(self,df):
        fig, axes = plt.subplots(len(df.columns)//3, 3, figsize=(16, 16))
        i = 0
        for triaxis in axes:
            for axis in triaxis:
                df.hist(column = df.columns[i], bins = 100, ax=axis)
                i = i+1
        return fig

    def data_update(self):
        self.valid_data.replace([3,4,5],0, inplace=True)
        self.valid_data.replace([6,7,8,9],1,inplace=True)
        


st.sidebar.markdown('# Exploratory data analysis')
st.markdown('## Reading the data')
dados = Dados()
st.write(dados.df.head())

st.markdown('## Getting the basic info from the data: ')

st.write(dados.basic_info(dados.df))

st.markdown('Function provided by [Leonardo Tavares](https://www.kaggle.com/leodaniel/)')

st.markdown('### Analyzing the quality distribution')

st.write(dados.quality_distribution(dados.df))

st.markdown('### Plotting the quality distribution')

st.write(dados.plot_hist_quality(dados.df))

st.markdown("### Plotting histograms for all the features")

st.write(dados.plot_hist_all(dados.df))

st.markdown('### Getting a description of the data')

st.write(dados.df.describe().T)

st.sidebar.markdown('Data Treatment')
st.markdown('# Data Treatment')
st.markdown('### Defining quality in two values:')
st.markdown('##### [0] for a Bad wine')
st.markdown('##### [1] for a Good wine')

st.markdown('### Displaying the quality column')
st.markdown('#### Old quality column:')
st.write(dados.df['quality'].head(30))
st.markdown('#### New quality column:')
st.write(dados.valid_data['quality'].head(30))
st.markdown('### New quality distribution')
st.write(dados.plot_hist_quality(dados.valid_data))

st.markdown("## Finally, let's have a look at the model's Data Drift")

html_path = f'{dados.path_to}/Extra Trees Classifier_Drift_Report_Classification.html'

# Web = req.get(f'{dados.path}/Extra Trees Classifier_Drift_Report_Classification.html')
# S = BeautifulSoup(Web.text, 'lxml')

import codecs
file = codecs.open(html_path, "r", "utf-8")

components.html(file.read(), scrolling=True, height=500)