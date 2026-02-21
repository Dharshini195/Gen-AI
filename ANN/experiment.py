import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
import pandas as pd

data = pd.read_csv("Churn_Modelling.csv")  #Reading CSV

data=data.drop(["RowNumber","CustomerId","Surname"],axis=1)  #Drop method is used to Drop (Delete permenantly) columns (or) rows -> here axis=1 means columns

label_encoder=LabelEncoder() #LabelEncoder used to encode a set of categorical values -> turns values into numbers (0s and 1s)
data["Gender"]=label_encoder.fit_transform(data["Gender"]) #Here Female turns 0 and Male turns 1

#One Hot Encoding -> Used to convert categorical values into vectors (numbers)
# We convert the Geography column into vectors using OneHotEncoder coz it has more than 2 values 

one_hot_encoder=OneHotEncoder()
one_hot_geo=one_hot_encoder.fit_transform(data[["Geography"]]) #Converting Geography column into vectors 
# print(one_hot_geo.toarray())  #Must use toarray() to get the correct value format


one_hot_geo_columns=one_hot_encoder.get_feature_names_out(["Geography"]) #Get the column names splitted by the encoder (3 columns as Geography had 3 unique values)
# print(one_hot_geo_columns)

new_geo_df = pd.DataFrame(one_hot_geo.toarray(),columns=one_hot_geo_columns) #Converting the encoded values as DF with the new encoded column names
# print(new_geo_df)

data=pd.concat([data.drop(["Geography"],axis=1),new_geo_df]) # Merging the old DF with the new encoded values for Geography by Droping the old Geography values (We only need encoded values)
# data=data.fillna("")
# print(data.head())

# with open("label_encoder_gender.pkl","wb") as file:
#     pickle.dump(label_encoder,file)

# with open("one_hot_encoder_geo.pkl","wb") as file:
#     pickle.dump(one_hot_encoder,file)

X=data.drop(["Exited"],axis=1)
y=data["Exited"]

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

with open("scaler.pkl","wb") as file:
    pickle.dump(scaler,file)
    
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import datetime

model=Sequential([
    Dense(64,activation='relu',input_shape=(X_train.shape[1],)), ## HL1 Connected wwith input layer
    Dense(32,activation='relu'), ## HL2
    Dense(1,activation='sigmoid')  ## output layer
]
 
)
 
model.summary()
 
import tensorflow
opt=tensorflow.keras.optimizers.Adam(learning_rate=0.01)
loss=tensorflow.keras.losses.BinaryCrossentropy()
 
## compile the model
model.compile(optimizer=opt,loss="binary_crossentropy",metrics=['accuracy'])
 
## Set up the Tensorboard
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
 
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callback=TensorBoard(log_dir=log_dir,histogram_freq=1)
 
 
## Set up Early Stopping
early_stopping_callback=EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
 
### Train the model
history=model.fit(
    X_train,y_train,validation_data=(X_test,y_test),epochs=100,
    callbacks=[tensorflow_callback,early_stopping_callback]
)
 
model.save('model.h5')
 

# How to log call in the vs code first open the terminal and run the comment
### tensorboard --logdir logs/fit
