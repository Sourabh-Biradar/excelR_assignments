import streamlit as st
import pandas as pd
import dill

with open("preprocessor.pkl", "rb") as f:
    preprocessor = dill.load(f)

with open("model.pkl", "rb") as f:
    model = dill.load(f)

st.title("Titanic Survival Predictor")

st.write("Enter passenger details below to predict survival:")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=32.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

if st.button("Predict Survival"):
    input_df = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked
    }])

    X_input = preprocessor.transform(input_df)

    prediction = model.predict(X_input)[0]

    if prediction == 1:
        st.success("Passenger would have SURVIVED!")
    else:
        st.error("Passenger would NOT have survived.")
