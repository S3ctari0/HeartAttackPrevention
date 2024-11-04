from flask import Flask, request, jsonify, render_template
import pickle
from flask_cors import CORS
import numpy as np

with open('best_model.pkl', 'rb') as archivo:
    modelo = pickle.load(archivo)

app = Flask(__name__)
CORS(app)

@app.route('/predecir', methods=['POST'])
def predecir():
    datos = request.get_json(force=True)
    entrada = np.array([
        datos['edad'],
        datos['dias_physic'],
        datos['dias_mental'],
        datos['historia'],
        datos['actividad_fisica'],
        datos['horas_sueño'],
        datos['dientes_removidos'],
        datos['ataque_cardiaco'],
        datos['angina'],
        datos['derrame_cerebral'],
        datos['asma'],
        datos['cancer_piel'],
        datos['epoc'],
        datos['depresivo'],
        datos['enfermedad_renal'],
        datos['dificultad_concentracion'],
        datos['artritis'],
        datos['diabetes'],
        datos['fumador'],
        datos['tc_tac'],
        datos['altura'],
        datos['peso'],
        datos['imc'],
        datos['alcohol_dias'],
        datos['dificilta_caminar'],
        datos['gripe_vacuna'],
        datos['neumonia_vacuna'],
        datos['tetanos_vacuna'],
        datos['covid19']
    ]).reshape(1, -1)
    
    prediccion = modelo.predict(entrada)
    
    return jsonify({'prediccion': prediccion[0]})

if __name__ == '__main__':
    app.run(debug=True)
