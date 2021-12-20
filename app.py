from flask import Flask, render_template , request , jsonify
from PIL import Image
import os , io , sys
import numpy as np 
import cv2
import base64
from numpy.lib.function_base import quantile

from numpy.lib.type_check import imag
from yolo_detection_images import runModel, run_model_tesseract

app = Flask(__name__)

############################################## THE REAL DEAL ###############################################
@app.route('/detect/image' , methods=['POST'])
def mask_image():
	# print(request.files , file=sys.stderr)
	file = request.files['image'].read() ## byte file
	npimg = np.fromstring(file, np.uint8)
	img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
	######### Do preprocessing here ################
	# img[img > 150] = 0
	## any random stuff do here
	################################################

	img = runModel(img)

	img = Image.fromarray(img.astype("uint8"))
	rawBytes = io.BytesIO()
	img.save(rawBytes, "JPEG")
	rawBytes.seek(0)
	img_base64 = base64.b64encode(rawBytes.read())
	return jsonify({'status':str(img_base64)})

@app.route('/detect' , methods=['POST'])
def mask_image_coordinates():
	file = request.files['image'].read() ## byte file
	npimg = np.fromstring(file, np.uint8)
	img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)

	try:
		predict = run_model_tesseract(img)
		kq = {}
		items = []
		for i in range(len(predict)):
			if(predict[i]['label'] != "item"):
				kq[predict[i]['label']] = predict[i]['text']
			else:
				name = ' '.join(predict[i]['text'].split(' ')[0 : -3])
				price = predict[i]['text'].split(' ')[-1]
				quantity = predict[i]['text'].split(' ')[-2]
				items.append({'name' : name,
					'price' : price,
					'quantity' : quantity})
		kq['items'] = items
	except:
		return jsonify({"Status" : "Can not detect image"})

	return jsonify(kq)

@app.route('/test' , methods=['GET'])
def test():
	print("log: got at test" , file=sys.stderr)
	return jsonify({'status':'succces'})
##################################################### THE REAL DEAL HAPPENS ABOVE ######################################

@app.route('/home')
def home():
	return render_template('./index.html')

@app.route('/')
def root():
	return jsonify({'api':
		[{'url' : 'https://tolbill.herokuapp.com/detect/image', 'form-data':'key: image, value: jpg/png/jpeg', 'method': 'POST', 'description' : 'return image predict in base64'},
			{'url' : 'https://tolbill.herokuapp.com/detect', 'form-data':'key: image, value: jpg/png/jpeg', 'method': 'POST', 'description' : 'return predict data in json'}
		]
	})

	
@app.after_request
def after_request(response):
    print("log: setting cors" , file = sys.stderr)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response


if __name__ == '__main__':
	app.run()
