from flask import Flask, request
from single_predict import Predict
from config import opt

app = Flask(__name__)

predict = Predict(opt)

@app.route("/calculate-joint-coordinates", methods=['POST'])
def calculate_joint_coordinates():
  frame = request.files['frame']
  centroid = request.json['centroid']

  predict.set_frame(frame, centroid)
  prediction = predict.execute()

  return { 'prediction': prediction }
		
if __name__ == '__main__':
   app.run(debug = True)