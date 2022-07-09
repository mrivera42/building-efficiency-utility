"""
@author: Max Rivera
"""
from flask import Flask, request, jsonify, render_template
import torch
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('template.html')

@app.route('/backend', methods=['POST'])
def predict():

    relative_compactness = request.form['relative_compactness']
    surface_area = request.form['surface_area']
    wall_area = request.form['wall_area']
    roof_area = request.form['roof_area']
    overall_height = request.form['overall_height']
    orientation = request.form['orientation']
    glazing_area = request.form['glazing_area']
    glazing_area_distribution = request.form['glazing_area_distribution']

    model = torch.load('models/model.pth')

    input_tensor = torch.Tensor([1,2,3,4,5,6,7,8])
    pred = model(input_tensor)
    print(pred)


    
    message = {'message': "hello there"}
    return jsonify(message)


if __name__ == "__main__":

    app.run()
