# lstm autoencoder recreate sequence
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, RepeatVector, TimeDistributed, GRU, Input

import pandas as pd
# define input sequence
#df=pd.read_excel('data/selected sample_Absorbance.xlsx',sheet_name='selected sample',index_col=0)
df=pd.read_csv('data/NAnderson2020MendeleyMangoNIRData.csv')
col = df.columns
#X=np.array(df[col[2:308]])
X=np.array(df[col[147:282]])
#Y=np.array(df[' Reference Value #1']/100)
Y=np.array(df['DM'])

samples = len(X)
timesteps_input = X.shape[1]
features = 1

# define model
model = Sequential()
model.add(Input(shape=(timesteps_input,)))
#model.add(LSTM(100,input_shape=(timesteps_input,features),return_sequences=False))
#model.add(SimpleRNN(100, return_sequences=False))
#model.add(RepeatVector(timesteps_output))
#model.add(LSTM(100, return_sequences=True))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(4))
model.add(Dense(1,activation='relu'))
model.compile(optimizer='adam', loss='mse')

#plot_model(model, show_shapes=True, to_file='reconstruct_lstm_autoencoder.png')
model.summary()
model.save('model/model_mango.h5')