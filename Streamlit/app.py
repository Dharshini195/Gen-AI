import pandas as pd
import numpy as np
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_data():
    #Loads Iris Dataset
    iris = load_iris() 

    #data -> (150,4) it has all the values of sepal length,widths & petal length,widths 
    #feature_names -> it is just the column names which are sepal length,widths & petal length,widths 
    df = pd.DataFrame(iris.data,columns=iris.feature_names) 

    #iris.target is the numeric representation of final target iris flowers which are 'setosa', 'versicolor', 'virginica', so the numeric is [0,1,2]
    df["species"] = iris.target

    #Returns the final df and iris flowers names ('setosa', 'versicolor', 'virginica')
    return df,iris.target_names

df,target_names = load_data()

model = RandomForestClassifier()

#Fitting model which X-AXIS as the full dataset of sepals & petals and Y-AXIS as numeric target (0,1,2)
model.fit(df.iloc[:,:-1],df["species"])


#Streamlit slider for choosing the input of sepal & petal values
st.sidebar.title("Select the Input Features")

sepal_length = st.sidebar.slider("Sepal Length",float(df["sepal length (cm)"].min()),float(df["sepal length (cm)"].max()))
sepal_width = st.sidebar.slider("Sepal Width",float(df["sepal width (cm)"].min()),float(df["sepal width (cm)"].max()))
petal_length = st.sidebar.slider("Petal Length",float(df["petal length (cm)"].min()),float(df["petal length (cm)"].max()))
petal_width = st.sidebar.slider("Petal Width",float(df["petal width (cm)"].min()),float(df["petal width (cm)"].max()))

#Storing the input values as Dataframe for prediction
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=df.columns[:-1])

#Passing our input data into Randomforest model for predicting the values
prediction = model.predict(input_data)

#Prediction results in numeric value of the species like (0,1,2), from there we are getting the top most prediction by 0th index. And using it to get the actual iris flower name
prediction_species = target_names[prediction[0]]

# Probabilities - Gives the probablities of the predicting values
probs = model.predict_proba(input_data)[0]
probs_df = pd.DataFrame(probs, index=target_names, columns=["Probability"])

st.title("Prediction Results")
st.write(f"🌸 The Predicted Iris is **{prediction_species}**")

st.subheader("Prediction Probabilities")
st.bar_chart(probs_df)