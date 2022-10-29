from re import I
import pandas as pd
import numpy as np
import pickle
import json

class IrisData():
    def __init__(self,a,b,c,d):
        self.SepalLengthCm=a
        self.SepalWidthCm=b
        self.PetalLengthCm=c
        self.PetalWidthCm=d
    def data(self):
        with open ('iris_model.pkl','rb')as f:
            self.model=pickle.load(f)
        with open ('iris_data.json','r')as f:
            self.data=json.load(f)
    def predict(self):
        self.data()
        array=np.zeros(len(self.data['Columns']))
        array[0]=self.SepalLengthCm
        array[1]=self.SepalWidthCm
        array[2]=self.PetalLengthCm
        array[3]=self.PetalWidthCm
        pred=self.model.predict([array])[0]
        print("Predicted class of the flower:-",pred)
        return pred
if __name__=="__main__":
    SepalLengthCm=5.1
    SepalWidthCm=3.5
    PetalLengthCm=1.4
    PetalWidthCm=0.2
    obj=IrisData(SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm)
    obj.predict()