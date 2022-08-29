import pandas as pd
import numpy as np

from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau,TensorBoard

from time import localtime, strftime

#df=pd.read_excel('data/selected sample_Absorbance.xlsx',sheet_name='selected sample',index_col=0)
df=pd.read_csv('data/NAnderson2020MendeleyMangoNIRData.csv')
col = df.columns
#X=np.array(df[col[2:308]])
X=np.array(df[col[147:282]])
#Y=np.array(df[' Reference Value #1']/100)
Y=np.array(df['DM'])


out = np.random.permutation(len(X))
X_shuf = X[out]
Y_shuf = Y[out]



splt = int(np.floor(len(X_shuf)*0.70))

X_train = X_shuf[:splt]
#X_train = np.expand_dims(X_train,axis=2)
Y_train = Y_shuf[:splt]

np.save('X_train.npy',X_train)
np.save('Y_train.npy',Y_train)

X_test = X_shuf[splt:]
#X_test = np.expand_dims(X_test,axis=2)
Y_test = Y_shuf[splt:]
np.save('X_test.npy',X_test)
np.save('Y_test.npy',Y_test)

Y_train = np.expand_dims(Y_train, axis=1)
Y_test = np.expand_dims(Y_test, axis=1)
#tensorflow.config.run_functions_eagerly(True)

model=load_model('model/model_mango_trained.h5',compile = False)

adam = optimizers.Adam(lr=0.00001)



model.compile(optimizer=adam, loss='mse')

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              patience=30)


history = model.fit(X_train, Y_train,
                    epochs=5000,
                    batch_size=64,
                    shuffle=True,
                    validation_data=(X_test, Y_test),
                    verbose=1,
                    callbacks=reduce_lr)

model.save('model/model_mango_trained.h5')

history = pd.DataFrame(history.history)

filename = 'history_'+strftime("%Y%m%d%H%M%S", localtime())
#filename = 'history_20200421164011.csv'

with open('history/'+filename+'.csv', 'a') as f:
    history = history.loc[pd.notnull(history.loss)] #remove NaN
    history.to_csv(f, header=True)

#plot loss

import matplotlib.pyplot as plt
df = history.loc[pd.notnull(history.loss)].iloc[:] #remove NaN

N=len(df["loss"])
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), df["loss"], label="train_loss")
plt.plot(np.arange(0, N), df["val_loss"], label="val_loss")
plt.legend(loc="upper right")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.savefig('history/'+filename+'.png')
plt.show()