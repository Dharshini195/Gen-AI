import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd

model=load_model("ANN\model.h5")

with open("ANN\label_encoder_gender.pkl","rb") as label:
    label_gender=pickle.load(label)

with open("ANN\one_hot_encoder_geo.pkl","rb") as one_hot:
    one_hot_geo=pickle.load(one_hot)

with open("ANN\scaler.pkl","rb") as scl:
    scalar=pickle.load(scl)

input_data={
    "CreditScore":600,
    "Geography" : "France",
    "Gender" : "Male",
    "Age" : 35,
    "Tenure" : 5,
    "Balance" : 60000,
    "NumOfProducts" : 2,
    "HasCrCard" : 1,
    "IsActiveMember" : 1,
    "EstimatedSalary" : 50000
}

geo_encoder=one_hot_geo.transform([[input_data["Geography"]]]).toarray()
geo_df=pd.DataFrame(geo_encoder,columns=one_hot_geo.get_feature_names_out(["Geography"]))
# print(geo_df)

input_df=pd.DataFrame([input_data])
input_df["Gender"]=label_gender.transform(input_df["Gender"])
input_df=pd.concat([input_df.drop("Geography",axis=1),geo_df],axis=1)
# print(input_df)

#Scalar values
input_scaled=scalar.transform(input_df)
# print(input_scaled)

#Prediction
output_predicted = model.predict(input_scaled)
print(output_predicted)

prediction_prob=output_predicted[0][0]
print(prediction_prob)

if prediction_prob>0.5:
    print("He/She is likely to churn")
else:
    print("No")