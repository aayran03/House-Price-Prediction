import streamlit as st
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from sklearn import datasets
import plotly.express as px

boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["MEDV"])

st.set_page_config(layout="wide")

#Sidebar navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select what you want to display:', ['Home', 'Data Summary', 'Data Header'])
model = pickle.load(open('model2.sav', 'rb'))

st.title('Boston House Price Prediction')
st.sidebar.header('Specify Input Parameters')
image = Image.open('hh.jpeg')
st.image(image, '')


def user_input_features():
    CRIM = st.sidebar.slider('CRIM', float(X.CRIM.min()), float(X.CRIM.max()), float(X.CRIM.mean()))
    ZN = st.sidebar.slider('ZN', float(X.ZN.min()), float(X.ZN.max()), float(X.ZN.mean()))
    INDUS = st.sidebar.slider('INDUS', float(X.INDUS.min()), float(X.INDUS.max()), float(X.INDUS.mean()))
    CHAS = st.sidebar.slider('CHAS', float(X.CHAS.min()), float(X.CHAS.max()), float(X.CHAS.mean()))
    NOX = st.sidebar.slider('NOX', float(X.NOX.min()), float(X.NOX.max()), float(X.NOX.mean()))
    RM = st.sidebar.slider('RM', float(X.RM.min()), float(X.RM.max()), float(X.RM.mean()))
    AGE = st.sidebar.slider('AGE', float(X.AGE.min()), float(X.AGE.max()), float(X.AGE.mean()))
    DIS = st.sidebar.slider('DIS', float(X.DIS.min()), float(X.DIS.max()), float(X.DIS.mean()))
    RAD = st.sidebar.slider('RAD', float(X.RAD.min()), float(X.RAD.max()), float(X.RAD.mean()))
    TAX = st.sidebar.slider('TAX', float(X.TAX.min()), float(X.TAX.max()), float(X.TAX.mean()))
    PTRATIO = st.sidebar.slider('PTRATIO', float(X.PTRATIO.min()), float(X.PTRATIO.max()), float(X.PTRATIO.mean()))
    B = st.sidebar.slider('B', float(X.B.min()), float(X.B.max()), float(X.B.mean()))
    LSTAT = st.sidebar.slider('LSTAT', float(X.LSTAT.min()), float(X.LSTAT.max()), float(X.LSTAT.mean()))
    data = {'CRIM': CRIM,
            'ZN': ZN,
            'INDUS': INDUS,
            'CHAS': CHAS,
            'NOX': NOX,
            'RM': RM,
            'AGE': AGE,
            'DIS': DIS,
            'RAD': RAD,
            'TAX': TAX,
            'PTRATIO': PTRATIO,
            'B': B,
            'LSTAT': LSTAT}
    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_features()

st.header('Specified Input parameters')
st.write(df)
st.write('---')

def data_summary():
    st.write("""The Boston Housing Dataset

The Boston Housing Dataset is a derived from information collected by the U.S. Census Service concerning housing in the area of Boston MA. The following describes the dataset columns:

CRIM - per capita crime rate by town


ZN - proportion of residential land zoned for lots over 25,000 sq.ft.


INDUS - proportion of non-retail business acres per town.


CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)


NOX - nitric oxides concentration (parts per 10 million)


RM - average number of rooms per dwelling


AGE - proportion of owner-occupied units built prior to 1940


DIS - weighted distances to five Boston employment centres


RAD - index of accessibility to radial highways


TAX - full-value property-tax rate per 10,000


PTRATIO - pupil-teacher ratio by town


B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town


LSTAT - % lower status of the population


MEDV - Median value of owner-occupied homes in $1000's""")


    st.header('Statistics of Dataframe')
    st.write(X.describe())



def data_header():
    st.header('Header of Dataframe')
    st.write(X.head())

if options == 'Home':
    price = model.predict(df)
    st.subheader('House Price')
    st.subheader('$' + str(np.round(price[0], 2)) + 'K')
elif options == 'Data Summary':
    data_summary()
elif options == 'Data Header':
    data_header()
elif options == 'Scatter Plot':
    displayplot()
elif options == 'Interactive Plots':
    interactive_plot()


hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
