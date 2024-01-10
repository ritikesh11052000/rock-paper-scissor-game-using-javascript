import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st


start = '2010-01-01'
end = '2022-12-31'


st.title('Stock Trand Prediction')

user_input = st.text_input('Enter Stock Ticker','TTM')
df = data.DataReader(user_input,'yahoo',start,end)

#Describing Data
st.subheader('data from 2010 - 2022')
st.write(df.describe())

#visulaizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close ,'b')
st.pyplot(fig)


# splitting Data into Training and Testing 

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler(feature_range=(0,1))

data_training_array = scalar.fit_transform(data_training)







#splitting data into x_train and y_train
#x_train = []
#y_train = []

#for i in range(100, data_training_Array.shape[0])):
  #  x_train.append(data_training_Array[i-100: i])
   # y_train.append(data_training_Array[i,0])

#x_train, y_train = np.array(x_train), np.Array(y_train)


#load my model
model = load_model('kears_model.h5')

#testing part

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True)
input_data = scalar.fit_transform(final_df)

x_train = []
y_train = []
for i in range(100, input_data.shape[0]):
    x_train.append(input_data[i - 100: i])
    y_train.append(input_data[i,0])

x_test , y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test, y_test)
scalar = scalar.scale_

scale_factor = 1/scalar[0]
y_predicted = y_predicted * scalar_factor
y_test = y_test * scale_factor



# final graph

st.subhreader('Predicted vs Orignal')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'orignal price')
plt.plot(y_predicted, 'r', label = 'pltedicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)