from flask import Flask, request, jsonify, render_template
import torch
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('template.html')


if __name__ == "__main__":

    app.run()
