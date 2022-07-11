"""
@author: Max Rivera
"""
from flask import Flask, request, jsonify, render_template
import torch

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('template.html')

@app.route('/backend', methods=['POST'])
def predict():

    # grab the input values from the API call 
    relative_compactness = float(request.form['relative_compactness'])
    surface_area = float(request.form['surface_area'])
    wall_area = float(request.form['wall_area'])
    roof_area = float(request.form['roof_area'])
    overall_height = float(request.form['overall_height'])
    orientation = float(request.form['orientation'])
    glazing_area = float(request.form['glazing_area'])
    glazing_area_distribution = float(request.form['glazing_area_distribution'])

    # create input tensor from input values 
    input_tensor = torch.Tensor([
        relative_compactness,
        surface_area,
        wall_area,
        roof_area,
        overall_height,
        orientation,
        glazing_area,
        glazing_area_distribution
    ])

    # load model 
    model = torch.jit.load('model_scripted.pt')
    model.to(torch.device('cpu'))
    model.eval()

    # get prediction results 
    pred = model(input_tensor)
    heating_load, cooling_load = pred[0].item(), pred[1].item()
    results = {
        'heating_load': heating_load,
        'cooling_load': cooling_load
    }

    return jsonify(results)


if __name__ == "__main__":

    app.run(host='0.0.0.0')  
