import streamlit as st

st.markdown("# Final Conclusions")
st.sidebar.markdown('# Final Conclusions')

st.markdown('### A - Data Analyzis and Manipulation')
st.markdown('Through data exploration, we could gather that the raw data consisted mostly of continuous values, with only one column of categorical values.')

st.markdown('##### Continuous Values')
st.markdown('Analyzing the graphics, we could not gather much, perhaps the data was a bit skewed, but not so much so that it would become a problem.')
st.markdown('##### Categorical Values')
st.markdown('Analyzing the categorical values, we gathered that it was divided into 7 values: [3, 4, 5, 6, 7, 8, 9]')
st.markdown('I decided to work on that, due to my lack of familiarity with several categorical values.')
st.markdown('Therefore, I reduced the values into two, separate categories: [0, 1]')
st.markdown('Respectively representing a a bad wine and a good wine.')

st.markdown('### B- Model Creation')
st.markdown('##### Creation')
st.markdown('With the data ready to be used in a model, I decided to research which model would best suit this data.')
st.markdown('For that, I used Pycaret in a jupyter notebook.')
st.markdown('The results were interesting, with ExtraTreesClassifier being the best model for this specific aplication.')
st.markdown('I then split the data 70/30 into train and test parts.')
st.markdown('Using Sklearn I fit the model and made the prediction.')
st.markdown('##### Metrics')
st.markdown("I started by looking at the model's score, but, following Leonardo's words, I knew that wasn't enough to evaluate the model.")
st.markdown('Therefore I proceeded to display and analyze several other metrics, and after a through inspection, I am satisfied with the results.')

st.markdown('# Final considerations')
st.markdown('In the future, I would like to take a more thorough approach into analyzing my continous variables, as I believe these results could be even further improved by that.')
st.markdown("I would like to thank Leonardo Tavares, for the amazing experience and opportunity provided, to learn data analyzis and machine learning.")
st.markdown("During this machine learning journey I learned a lot and had an amazing experience, so thank you!")
st.markdown('#### And finally, I would like to thank you for your time reading thus far, whoever you are!')
st.markdown('## Thank you for taking part in another step of my journey through machine learning!')
