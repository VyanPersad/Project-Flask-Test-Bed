from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
 
app = Flask(__name__)
 
upload_folder = os.path.join('static', 'uploads')
 
app.config['UPLOAD'] = upload_folder
 
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD'], filename))
        img = os.path.join(app.config['UPLOAD'], filename)
        img = cv2.imread(img)
        lower = np.array([3, 15, 10], dtype="uint8")
        upper = np.array([20, 255, 255], dtype="uint8")
        image = cv2.resize(img, (300, 300))
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(converted, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skinMask = cv2.erode(skinMask, kernel, iterations=2)
        skinMask = cv2.dilate(skinMask, kernel, iterations=2)
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        skin = cv2.bitwise_and(image, image, mask=skinMask)
        contours, _ = cv2.findContours(skinMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        non_black_boxes = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if cv2.contourArea(contour) > 100:
                non_black_boxes.append((x, y, w, h))
            
        skin_cropped = skin.copy()
        for box in non_black_boxes:
            x, y, w, h = box
            #img output,upper left, lower right, BGR Color, thickness
            cv2.rectangle(skin_cropped, (x, y), (x + w, y + h), (0, 255, 0), 2)

        largest_contour = max(contours, key=cv2.contourArea)
        largest_contour_mask = np.zeros_like(skinMask)
        cv2.drawContours(largest_contour_mask, [largest_contour], 0, 255, thickness=cv2.FILLED)
        largest_contour_image = cv2.bitwise_and(image, image, mask=largest_contour_mask)
        
        #This specifically writes the image to a file called skin1.png
        cv2.imwrite('static/skin1.png',largest_contour_image)
                
    return render_template('index.html', 
                               img_path1='skin1.png',
                               img_path2='')
 
 
if __name__ == '__main__':
    app.run(debug=True, port=5000)