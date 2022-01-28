import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

df = pd.read_csv('textures_data.csv', sep=',')
data = df.to_numpy()

x = data[:,:-1].astype('float')
y = data[:,-1]

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(y)

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

x_train, x_test, y_train, y_test = train_test_split(x, onehot_encoded, test_size=0.3)

model = Sequential()
model.add(Dense(10, input_dim=72, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, epochs=100, batch_size=10, shuffle=True)

y_pred = model.precict(x_test)
y_pred_int = np.argmax(y_pred, axis=1)
y_test_int = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test_int, y_pred_int)
print(cm)
