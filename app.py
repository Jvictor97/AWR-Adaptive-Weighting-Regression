from flask import Flask, request
from single_predict import Predict
from flask_ngrok import run_with_ngrok
from config import opt
import numpy as np

app = Flask(__name__)
run_with_ngrok(app)

predict = Predict(opt)

@app.route("/calculate-joint-coordinates", methods=['POST'])
def calculate_joint_coordinates():
  frame = np.array(request.json['frame'])
  centroid = np.array(request.json['centroid'])
  is_left_hand = request.json['is_left_hand']

  predict.set_frame(frame, centroid, is_left_hand)
  prediction = predict.execute()

  return { 'prediction': prediction.tolist() }
		
if __name__ == '__main__':
   app.run()