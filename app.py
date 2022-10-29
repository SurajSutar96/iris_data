
from flask import Flask,render_template,request,jsonify,url_for
app=Flask(__name__)
from util import IrisData
import numpy as np
import pickle
with open('iris_model.pkl','rb')as f:
    model=pickle.load(f)
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    ip_features = [float(x) for x in request.form.values()]
    features = [np.array(ip_features)]
    pred=model.predict(features)[0]
    return render_template("home.html", prediction_text = "Category is {}".format(pred))
if __name__=="__main__":
    app.run(port='3333',debug=True)