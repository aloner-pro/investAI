from keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load the model
loaded_model = load_model("my_model.keras")

def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)

df = pd.read_csv('file.csv')
df['Date'] = pd.to_datetime(df['Date'])
dfc = df.reset_index()['Close']
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(dfc).reshape(-1,1))

training_size=int(len(df1)*0.65)
test_data=df1[training_size:len(df1),:1]
time_step = 100
X_test, ytest = create_dataset(test_data, time_step)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

test_predict=loaded_model.predict(X_test)
test_predict=scaler.inverse_transform(test_predict)
print(test_predict[-5:])
predictions_list = test_predict[-5:].flatten().tolist()

# Format the string
predictions_str = ", ".join(map(str, predictions_list))
output_str = f"The stock prices for the next 5 days that has been predicted via LSTM Model are {predictions_str}  : values are in  INR"

# Print the formatted string
print(output_str)

day_new=np.arange(1,101)
day_pred=np.arange(101,131)
df3=df1.tolist()

print("Test", len(test_data))
x_input=test_data[200:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()
lst_output=[]
n_steps=100
i=0
while(i<30):

    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        # print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = loaded_model.predict(x_input, verbose=0)
        # print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = loaded_model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1


df3.extend(lst_output)
df3=scaler.inverse_transform(df3).tolist()
