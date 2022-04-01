from flask import Flask, request
from single_predict import Predict
from config import opt

# app = Flask(__name__)

# @app.route("/calculate-joint-coordinates", methods=['POST'])
# def calculate_joint_coordinates():
#   body = request.json

#   return { 'foo': 123 }, 201

p1 = Predict(opt, 'http')
p1.set_frame([[]], [1,2,3])

print(p1)