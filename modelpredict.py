import pandas as pd
import numpy as np
from keras.models import load_model
from time import localtime, strftime

#df=pd.read_excel('data/selected sample_Absorbance.xlsx',sheet_name='selected sample',index_col=0)
#df=pd.read_csv('data/selected sample_Absorbance_cut.csv',index_col=0)
#col = df.columns
#X=np.array(df[col[147:282]])
#X = np.expand_dims(X,axis=2)
#Y=np.array(df[' Reference Value #1']/100)
X = np.load('X_test.npy')
Y = np.load('Y_test.npy')

model=load_model('model/model_mango_trained.h5')

Y_pred = model.predict(X)
Y_pred = np.squeeze(Y_pred)

#Y_pred = np.squeeze(Y_pred)
#now add them to our data frame
#df['y_pred'] = Y_pred





from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error,mean_absolute_percentage_error
from math import sqrt

#Y_pred = df['y_pred']


# The mean absolute error
print("Mean Absolute Error %.7f" % mean_absolute_error(Y, Y_pred))
# The mean squared error
print("Root Mean squared error: %.7f" % sqrt(mean_squared_error(Y, Y_pred)))
# Explained variance score: 1 is perfect prediction
print('Variance score %.7f' % r2_score(Y, Y_pred))



#save prediction result
filename = 'predicted'+strftime("%Y%m%d%H%M%S", localtime())
import matplotlib.pyplot as plt
#df.to_csv('predicted/'+filename+'.csv', header=True)
plt.style.use("ggplot")
plt.figure()
plt.scatter(Y, Y_pred)
plt.plot(Y, Y, '-g', label='Expected regression line')
plt.title("Actual vs Pred")
plt.xlabel("Actual")
plt.ylabel("Pred")
#plt.savefig('pred/'+filename+'.png')
plt.show()
