import streamlit as st
import joblib
import pandas as pd 
# Load model and encoders
model = joblib.load("best_model.pkl")
LEC_Sex = joblib.load("LEC_Sex.pkl")
LEC_Embarked = joblib.load("LEC_Embarked.pkl")

st.title('⛴️ Titanic Survival Prediction App')
st.text('Provide the passenger details and click on Predict')

# User Input Field
pclass = st.selectbox("Passenger Class",[1,2,3])
Sex = st.selectbox("Sex",["male","female"])
Age = st.number_input("Age",0,100,30)
SibSp = st.number_input("SibSp",0,10,0)
Parch = st.number_input("Parch",0,10,0)
Fare = st.number_input("Fare",0.0,600.0,32.0)
Embarked = st.selectbox("Embarked",["C","Q","S"])

# Encode User Input
Sex_encoded = LEC_Sex.transform([Sex])[0]
Embarked_encoded = LEC_Embarked.transform([Embarked])[0]

# Create Input DataFrame
Input_data = pd.DataFrame([{
    "Pclass" : pclass,
    "Sex": Sex_encoded,
    "Age": Age,
    "SibSp": SibSp,
    "Parch":Parch,
    "Fare":Fare,
    "Embarked": Embarked_encoded
}])

if st.button("Predict"):
    Prediction = model.predict(Input_data)[0]
    if Prediction == 1:
        st.success("The Person Survived")
    else:
        st.error("The Person Did Not Survive")
