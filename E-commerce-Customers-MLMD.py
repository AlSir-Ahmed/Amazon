import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.image("./E-commerce Customers.jpg")
st.write("""
# Amazon Revenues Prediction App

This app predicts the revenues of Amazon!
""")
st.write("---")

amazon = pd.read_csv("Ecommerce Customers1.csv")
X = amazon.drop("Yearly Amount Spent", axis = 1)
Y = amazon["Yearly Amount Spent"]

st.sidebar.header("Specify Input Parameters")
st.sidebar.write("---")
def user_input_features():
    ASL = st.sidebar.slider("Avg. Session Length", X["Avg. Session Length"].min(), X["Avg. Session Length"].max())
    TOA = st.sidebar.slider("Time on App", X["Time on App"].min(), X["Time on App"].max())
    TOW = st.sidebar.slider("Time on Website", X["Time on Website"].min(), X["Time on Website"].max())
    LOM = st.sidebar.slider("Length of Membership", X["Length of Membership"].min(), X["Length of Membership"].max())
    data = {"Avg. Session Length": ASL,
            "Time on App": TOA,
            "Time on Website": TOW,
            "Length of Membership": LOM}
    features = pd.DataFrame(data, index = [0])
    return features
df = user_input_features()
st.header("Specified Input Parameters")
st.write(df)
st.write("---")

model = LinearRegression()
model.fit(X, Y)
prediction = model.predict(df)
st.header("Prediction of revenues")
st.write(prediction)
