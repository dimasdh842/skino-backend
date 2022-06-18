from flask import Flask,request
from flask_restful import Resource,Api
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

import json

model = tf.keras.models.Sequential()
def train_model():
    data = pd.read_csv('./dataset.csv')
    
    x = data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]].values
    y = data.iloc[:,-1].values

    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,train_size=0.7,random_state=42)

    
    model.add(tf.keras.Input(shape=(19,)))
    model.add(tf.keras.layers.Dense(units=4,activation=tf.keras.activations.hard_sigmoid, use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',))
    model.add(tf.keras.layers.Dense(units=1,activation=tf.keras.activations.hard_sigmoid, use_bias=True,kernel_initializer='glorot_uniform',bias_initializer='zeros',))
    model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(0.1))
    model.fit(xtrain,ytrain,epochs=1000)

# init objek flask
app = Flask(__name__)
api = Api(app)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['CORS_ORIGINS'] = '*'
identity = {}
train_model()

class Controller(Resource) :
    def get(self):
        return identity
    def post(self):
        data = request.get_json()
        print(data['data'])
        identity["result"] = {"result":json.dumps(str(model.predict([data['data']]) [0] [0]))}

        response = {"msg":"data terikirim"}
        return response

api.add_resource(Controller,"/api",methods=["GET","POST"])

if __name__ == "__main__":
    app.run(debug=True, port=5001)