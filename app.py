import streamlit as st
import pandas as pd
import pickle

with open('model.pkl', 'rb') as model_file:
    clf_iris = pickle.load(model_file)

with open('stand.pkl', 'rb') as scaler_file:
    scaler_iris = pickle.load(scaler_file)


html_attribution = """
    <div style="background-color:#28a745;padding:20px;margin-bottom:20px">
    <p style="color:white;text-align:center;font-size:22px;">Developed by Pruthvik Machhi</p>
    </div>
    """
st.markdown(html_attribution, unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)





html_temp_subtitle = """
    <div style="background-color:#007bff;padding:10px;margin-bottom:20px">
    <h2 style="color:white;text-align:center;">Iris Flower Prediction</h2>
    </div>
    """
st.markdown(html_temp_subtitle, unsafe_allow_html=True)

def user_input_features():
    sepal_length = st.number_input('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.number_input('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.number_input('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.number_input('Petal width', 0.1, 2.5, 0.2)
    data = {'SepalLengthCm': sepal_length,
            'SepalWidthCm': sepal_width,
            'PetalLengthCm': petal_length,
            'PetalWidthCm': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

st.subheader('Enter Input ')
df = user_input_features()

st.subheader(' Input parameters')
st.write(df)

expected_features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
df = df[expected_features]

if st.button('Predict'):
    scaled_features = scaler_iris.transform(df)
    prediction = clf_iris.predict(scaled_features)
    species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    predicted_species = species[prediction[0]]
    st.subheader('Prediction')
    st.write(f"The predicted Iris species is: **{predicted_species}**")

