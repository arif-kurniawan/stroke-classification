from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

model_file = open('model-arif.pkl', 'rb')
model = pickle.load(model_file, encoding='bytes')


@app.route('/')
def index():
    return render_template('index.html', hasil=0)

@app.route('/predict', methods=['POST'])
def predict():
    jenis_kelamin=request.form['jenis_kelamin']
    umur=float(request.form['umur'])
    tekanan_darah_tinggi=request.form['tekanan_darah_tinggi']
    penyakit_jantung=request.form['penyakit_jantung']
    pernah_menikah=request.form['pernah_menikah']
    pekerjaan=request.form['pekerjaan']
    tempat_tinggal=request.form['tempat_tinggal']
    gula_darah=float(request.form['gula_darah'])
    bmi=int(request.form['bmi'])
    merokok=request.form['merokok']

    x=np.array([[jenis_kelamin,umur,tekanan_darah_tinggi,penyakit_jantung,pernah_menikah,pekerjaan,tempat_tinggal,gula_darah,bmi,merokok]])
    print(x)
    prediction=model.predict(x)
    output = prediction[0]

    return render_template('index.html', hasil=output)

if __name__ == '__main__':
    app.run(debug=True)
