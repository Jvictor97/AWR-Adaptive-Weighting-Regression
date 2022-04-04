from flask import Flask, request
from single_predict import Predict
from flask_ngrok import run_with_ngrok
from config import opt

app = Flask(__name__)
run_with_ngrok(app)

predict = Predict(opt)

@app.route("/calculate-joint-coordinates", methods=['POST'])
def calculate_joint_coordinates():
  frame = request.json['frame']
  centroid = request.json['centroid']

  predict.set_frame(frame, centroid)
  prediction = predict.execute()

  return { 'prediction': prediction }
		
if __name__ == '__main__':
   app.run()