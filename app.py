from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from skimage.feature import graycomatrix
from skimage.feature import graycoprops
from skimage.filters import gabor
import os
import cv2
import numpy as np
import math
import csv
import pandas as pd
#import tensorflow.compat.v1 as tf
import tensorflow as tf 
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
 
upload_folder = os.path.join('static', 'uploads')
app.config['UPLOAD'] = upload_folder

model_path_2 = "static/model/my_diabetes_BM.h5"
model_2 = load_model(model_path_2)

def sRGBtoLinearRGB(c):
    if c <= 0.04045:
        return c / 12.92
    else:
        return ((c + 0.055) / 1.055) ** 2.4

def rgb_to_cmyk(r, g, b):
    if (r, g, b) == (0, 0, 0):
        k=1
        # black
        return 0, 0, 0, 100

    # rgb [0,255] -> cmy [0,1]
    c = 1 - (r / RGB_SCALE)
    m = 1 - (g / RGB_SCALE)
    y = 1 - (b / RGB_SCALE)

    # extract out k [0, 1]
    min_cmy = min(c, m, y)
    c = round((c - min_cmy) / (1 - min_cmy) * 100)
    m = round((m - min_cmy) / (1 - min_cmy) * 100)
    y = round((y - min_cmy) / (1 - min_cmy) * 100)
    k = round(min_cmy)

    # rescale to the range [0,CMYK_SCALE]
    return int(c), int(m), int(y), int(k) 

def rgbToLab(r, g, b) :
    r = r / 255
    g = g / 255
    b = b / 255

    if r > 0.04045:
        r = (r + 0.055) / 1.055 ** 2.4
    else:
        r = r / 12.92

    if g > 0.04045:
        g = (g + 0.055) / 1.055 ** 2.4
    else:
        g = g / 12.92

    if b > 0.04045:
        b = (b + 0.055) / 1.055 ** 2.4
    else:
        b = b / 12.92

    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883

    if x > 0.008856:
        x = x ** (1 / 3)
    else:
        x = (7.787 * x) + 16 / 116
    if y > 0.008856:
        y = y ** (1 / 3)
    else:
        y = (7.787 * y) + 16 / 116
    if z > 0.008856:
        z = z ** (1 / 3)
    else:
        z = (7.787 * z) + 16 / 116

    return f"{(116 * y) - 16:.5f},{500 * (x - y):.5f},{200 * (y - z):.5f}"

def rgbToHsv(r, g, b):
    rabs = r / 255
    gabs = g / 255
    babs = b / 255
    v = max(rabs, gabs, babs)
    diff = v - min(rabs, gabs, babs)
    diffc = lambda c: (v - c) / 6 / diff + 1 / 2
    percentRoundFn = lambda num: round(num * 100) / 100
    if diff == 0:
        h = s = 0
    else:
        s = diff / v
        rr = diffc(rabs)
        gg = diffc(gabs)
        bb = diffc(babs)
        if rabs == v:
            h = bb - gg
        elif gabs == v:
            h = (1 / 3) + rr - bb
        elif babs == v:
            h = (2 / 3) + gg - rr
        if h < 0:
            h += 1
        elif h > 1:
            h -= 1
    return f"{round(h * 360)},{percentRoundFn(s * 100)},{percentRoundFn(v * 100)}"

def rgbToLuminance(r, g, b):
      return round((((0.2126*r/255)+(0.7152*g/255)+(0.0722*b/255))*100),3)

def temperature2rgb(kelvin):
    temperature = kelvin / 100.0
    if temperature < 66.0:
        red = 255
    else:
        red = temperature - 55.0
        red = 351.97690566805693 + 0.114206453784165 * red - 40.25366309332127 * math.log(red)
        if red < 0:
            red = 0
        if red > 255:
            red = 255
    if temperature < 66.0:
        green = temperature - 2
        green = -155.25485562709179 - 0.44596950469579133 * green + 104.49216199393888 * math.log(green)
        if green < 0:
            green = 0
        if green > 255:
            green = 255
    else:
        green = temperature - 50.0
        green = 325.4494125711974 + 0.07943456536662342 * green - 28.0852963507957 * math.log(green)
        if green < 0:
            green = 0
        if green > 255:
            green = 255
    if temperature >= 66.0:
        blue = 255
    else:
        if temperature <= 20.0:
            blue = 0
        else:
            blue = temperature - 10
            blue = -254.76935184120902 + 0.8274096064007395 * blue + 115.67994401066147 * math.log(blue)
            if blue < 0:
                blue = 0
            if blue > 255:
                blue = 255
    return {"red": round(red), "blue": round(blue), "green": round(green)}

def rgbToTemperature(r, g, b):
    epsilon = 0.4
    minTemperature = 1000
    maxTemperature = 40000
    while maxTemperature - minTemperature > epsilon:
        temperature = (maxTemperature + minTemperature) / 2
        testRGB = temperature2rgb(temperature)
        if (testRGB["blue"] / testRGB["red"]) >= (b / r):
            maxTemperature = temperature
        else:
            minTemperature = temperature
    return round((temperature),3)

def rgbToRyb(r, g, b):
    # Remove the whiteness from the color.
    w = min(r, g, b)
    r -= w
    g -= w
    b -= w

    mg = max(r, g, b)

    # Get the yellow out of the red+green.
    y = min(r, g)
    r -= y
    g -= y

    # If this unfortunate conversion combines blue and green, then cut each in
    # half to preserve the value's maximum range.
    if b and g:
        b /= 2.0
        g /= 2.0

    # Redistribute the remaining green.
    y += g
    b += g

    # Normalize to values.
    my = max(r, y, b)
    if my:
        n = mg / my
        r *= n
        y *= n
        b *= n

    # Add the white back in.
    r += w
    y += w
    b += w

    # And return back the ryb typed accordingly.
    return str(int(r)) + "," + str(int(y)) + "," + str(int(b))

def rgbToXyz(r, g, b):
    r = sRGBtoLinearRGB(r / 255)
    g = sRGBtoLinearRGB(g / 255)
    b = sRGBtoLinearRGB(b / 255)

    X = 0.4124 * r + 0.3576 * g + 0.1805 * b
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    Z = 0.0193 * r + 0.1192 * g + 0.9505 * b

    return str(int(X * 100)) + "," + str(int(Y * 100)) + "," + str(int(Z * 100))

#0<--Blk+++White-->255
RGB_SCALE = 255
CMYK_SCALE = 100

def get_pigmentation_info(r, g, b):
    Pig = []
    Pig.append(rgb_to_cmyk(r, g, b))
    Pig.append(rgbToLab(r, g, b))
    Pig.append(rgbToHsv(r, g, b))
    Pig.append(rgbToLuminance(r, g, b))
    Pig.append(rgbToTemperature(r, g, b))
    Pig.append(rgbToRyb(r, g, b))
    Pig.append(rgbToXyz(r, g, b))
    
    Pig.reverse()

    return Pig


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    
    file_path = 'static/model/trimmed_data_2.csv'
    df = pd.read_csv(file_path)
    scaler =""
    X = df[['Contrast',	'Dissimilarity',	'Homogeneity',	'Energy',	'Correlation',	'ASM','Gabor1','Gabor2','Gabor2','Gabor4','Gabor5','Gabor6','Gabor7','Gabor8','Gabor9','Gabor10','Gabor11','k-Value']].values
    y = df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train.shape
    X_test.shape
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    #model_2.fit(x=X_train, y=y_train, epochs=250, verbose=0)

    cont = ""
    diss = ""
    homo = ""
    ener = ""
    corr = ""
    asm = ""
    g1=""
    g2=""
    g3=""
    g4=""
    g5=""
    g6=""
    g7=""
    g8=""
    g9=""
    g10=""
    g11=""
    k_value=""
    prediction = ""
    outcome = ""
    
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
        
        file_path = 'static/cropped.png'
        cv2.imwrite(file_path,largest_contour_image)
        
        img = cv2.imread(file_path)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        #Calculate GLCM with specified parameters
        distances = [1]  # Distance between pixels
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Angles for pixel pairs
        levels = 256  # Number of gray levels
        symmetric = True
        normed = True
        
        glcm = graycomatrix(gray_image, distances, angles, levels=levels, symmetric=symmetric, normed=normed)        
        
        cont = round(graycoprops(glcm, 'contrast').ravel()[0], 4)
        diss = round(graycoprops(glcm, 'dissimilarity').ravel()[0], 4)
        homo = round(graycoprops(glcm, 'homogeneity').ravel()[0], 4)
        ener = round(graycoprops(glcm, 'energy').ravel()[0], 4)
        corr = round(graycoprops(glcm, 'correlation').ravel()[0], 4)
        asm = round(graycoprops(glcm, 'ASM').ravel()[0], 4)

        frequencies = [0.1, 0.3, 0.5]
        kernels = []

        # Generate Gabor filter kernels
        for frequency in frequencies:
            for theta in angles:
                kernel = np.real(gabor(gray_image, frequency, theta=theta))
                kernels.append(np.mean(kernel))

        # Convert the list of Gabor features to a numpy array
        gabor_features = np.array(kernels)
        g1 = round(gabor_features[0],4)
        g2 = round(gabor_features[1],4)
        g3 = round(gabor_features[2],4)
        g4 = round(gabor_features[3],4)
        g5 = round(gabor_features[4],4)
        g6 = round(gabor_features[5],4)
        g7 = round(gabor_features[6],4)
        g8 = round(gabor_features[7],4)
        g9 = round(gabor_features[8],4)
        g10 = round(gabor_features[9],4)
        g11 = round(gabor_features[10],4)

        # Convert the cropped image to CMYK color space manually
        b, g, r = cv2.split(img)
        c = 255 - r
        m = 255 - g
        y = 255 - b
        k = np.minimum(np.minimum(c, m), y)
        c = cv2.subtract(c, k)
        m = cv2.subtract(m, k)
        y = cv2.subtract(y, k)
        # Compute the average value of the K channel
        k_value = np.mean(k)
        #X = df[['Dissimilarity','Contrast','Gabor6','Gabor7','Gabor8','Gabor9','Gabor10','Gabor11']].values

        #X = df[['Contrast',	'Dissimilarity',	'Homogeneity',	'Energy',	'Correlation',	'ASM',
        #'Gabor1','Gabor2','Gabor2','Gabor4','Gabor5','Gabor6','Gabor7','Gabor8','Gabor9','Gabor10','Gabor11','k-Value']].values
        patient = np.array([
            [cont,diss,homo,ener,corr,asm,
             g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11,
             k_value]])
        patient_scaled = scaler.transform(patient)
        prediction = model_2.predict(patient_scaled).tolist() 

        prediction = prediction[0][0]
        if prediction >= 0.5:
            outcome = "Potentially Positive"
        elif prediction < 0.5:
            outcome = "Potentially Negative"

    
    return render_template('index.html', 
                                img_path1='cropped.png',
                                outcome = outcome)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    file_path = 'static\model\diabetes.csv'
    df = pd.read_csv(file_path)
    scaler =""
    X = df[['Glucose','Insulin','SkinThickness','BMI']].values
    y = df['Outcome'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train.shape
    X_test.shape
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    model.fit(x=X_train, y=y_train, epochs=250, verbose=0)

    glucose = ""
    insulin = ""
    skinThicc = ""
    bmi = ""
    prediction = ""

    if request.method == 'POST':
        glucose = float(request.form['glucose_In'])
        insulin = float(request.form['insulin_In'])
        skinThicc = float(request.form['skinthicc_In'])
        bmi = float(request.form['bmi_In'])
        
        patient = np.array([[glucose,insulin,skinThicc,bmi]])
        patient_scaled = scaler.transform(patient)
        prediction = model.predict(patient_scaled).tolist() 

        prediction = prediction[0][0]

    return render_template('predict.html', 
                           glucose=glucose,
                           insulin=insulin,
                           skinThicc=skinThicc,
                           bmi=bmi,
                           prediction=prediction
                           )

if __name__ == '__main__':
    app.run(debug=True, port=5000)