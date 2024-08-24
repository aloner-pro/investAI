import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import globals  # Import the global module


# Define a function to load and cache the model
@st.cache_resource
def load_trained_model():
    return load_model("my_model.keras")


# Define a function to prepare the data
def prepare_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    dfc = df.reset_index()['Close']

    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(dfc).reshape(-1, 1))

    return df, df1, scaler


# Define a function to create datasets
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# Load the trained model
model = load_trained_model()

# Streamlit app
st.title("Stock Price Prediction")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df, df1, scaler = prepare_data(uploaded_file)

    # Prepare test data
    time_step = 100
    X_test, y_test = create_dataset(df1, time_step)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Predict using the model
    test_predict = model.predict(X_test)
    test_predict = scaler.inverse_transform(test_predict)

    # Display the last 5 predictions
    st.write("Predicted stock prices for the next 5 days:")
    st.write(test_predict[-5:])

    # Generate future predictions
    day_new = np.arange(1, 101)
    day_pred = np.arange(101, 131)
    df3 = df1.tolist()

    x_input = X_test[-1].reshape(1, -1)
    temp_input = x_input[0].tolist()
    lst_output = []
    n_steps = 100
    i = 0

    while i < 30:
        if len(temp_input) > n_steps:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, n_steps, 1)
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i += 1
        else:
            x_input = np.array(temp_input).reshape(1, n_steps, 1)
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i += 1

    df3.extend(lst_output)
    df3 = scaler.inverse_transform(np.array(df3).reshape(-1, 1)).tolist()

    # Create the output string
    output_str = f"The predicted stock price for the next 5 days is: {df3[-5:]}"

    # Update the global variables
    globals.update_globals(df3, output_str)
