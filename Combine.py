import cv2
import numpy as np
import random
import math
from skimage.feature import graycomatrix, graycoprops

RGB_SCALE = 255
CMYK_SCALE = 100

def sRGBtoLinearRGB(c):
    if c <= 0.04045:
        return c / 12.92
    else:
        return ((c + 0.055) / 1.055) ** 2.4

def rgb_to_cmyk(rgb):
    r, g, b = rgb
    if (r, g, b) == (0, 0, 0):
        return 0, 0, 0, CMYK_SCALE

    c = 1 - r / RGB_SCALE
    m = 1 - g / RGB_SCALE
    y = 1 - b / RGB_SCALE

    min_cmy = min(c, m, y)
    c = (c - min_cmy) / (1 - min_cmy)
    m = (m - min_cmy) / (1 - min_cmy)
    y = (y - min_cmy) / (1 - min_cmy)
    k = min_cmy

    return int(c * CMYK_SCALE), int(m * CMYK_SCALE), int(y * CMYK_SCALE), int(k * CMYK_SCALE)

def rgbToLab(rgb):
    r, g, b = rgb
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

def rgbToHsv(rgb):
    r, g, b = rgb
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

def rgbToLuminance(rgb):
    r, g, b = rgb
    return (((0.2126*r/255)+(0.7152*g/255)+(0.0722*b/255))*100)

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

def rgbToTemperature(rgb):
    epsilon = 0.4
    minTemperature = 1000
    maxTemperature = 40000
    while maxTemperature - minTemperature > epsilon:
        temperature = (maxTemperature + minTemperature) / 2
        testRGB = temperature2rgb(temperature)
        if (testRGB["blue"] / testRGB["red"]) >= (rgb[2] / rgb[0]):
            maxTemperature = temperature
        else:
            minTemperature = temperature
    return round(temperature)

def rgbToRyb(rgb):
    r, g, b = rgb
    w = min(r, g, b)
    r -= w
    g -= w
    b -= w

    mg = max(r, g, b)

    y = min(r, g)
    r -= y
    g -= y

    if b and g:
        b /= 2.0
        g /= 2.0

    y += g
    b += g

    my = max(r, y, b)
    if my:
        n = mg / my
        r *= n
        y *= n
        b *= n

    r += w
    y += w
    b += w

    return str(int(r)) + "," + str(int(y)) + "," + str(int(b))

def rgbToXyz(rgb):
    r, g, b = rgb
    r = sRGBtoLinearRGB(r / 255)
    g = sRGBtoLinearRGB(g / 255)
    b = sRGBtoLinearRGB(b / 255)

    X = 0.4124 * r + 0.3576 * g + 0.1805 * b
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    Z = 0.0193 * r + 0.1192 * g + 0.9505 * b

    return str(int(X * 100)) + "," + str(int(Y * 100)) + "," + str(int(Z * 100))

def getGLCMFeatures(img):
    distances = [1]  # you can define multiple distances if needed
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # define angles for co-occurrence matrix
    glcm = graycomatrix(img, distances, angles, levels=256, symmetric=True, normed=True)
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
    features = np.hstack([graycoprops(glcm, prop).ravel() for prop in properties])
    return features

def getPixels(file_path):
    img = cv2.imread(file_path)  # Read the image
    if img is None:
        print("Error: Unable to read the image.")
        return

    print("Image shape:", img.shape)  # Debug print to check image shape
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    imgW = np.size(img,1)
    imgH = np.size(img,0)

    randoW = []
    randoH = []
    pxlArray = []
    grayPixels = []

    count = 0
    n = 10
    while count != n: 
        for i in range(0,n):
            h = random.randint(1,imgH-1)
            w = random.randint(1,imgW-1)
            if(np.any(img[h,w])!=0):
                randoW.append(w)
                randoH.append(h)
                count = count + 1
            else:
                break

    for i in range(0,n):
        pxlArray.append(img[randoH[i], randoW[i]])
        grayPixels.append(int(sum(pxlArray[i])//3))

    minPxVal = np.argmin(grayPixels)
    maxPxVal = np.argmax(grayPixels)

    r_min, g_min, b_min = pxlArray[minPxVal]
    r_max, g_max, b_max = pxlArray[maxPxVal]

    print("CMYK -","Hyper ", rgb_to_cmyk((r_min, g_min, b_min)),"       Normal ",rgb_to_cmyk((r_max, g_max, b_max)))
    print("Lab  -","Hyper ", rgbToLab((r_min, g_min, b_min)),"          Normal ",rgbToLab((r_max, g_max, b_max)))
    print("HSV  -","Hyper ", rgbToHsv((r_min, g_min, b_min)),"          Normal ",rgbToHsv((r_max, g_max, b_max)))
    print("LUM  -","Hyper ", rgbToLuminance((r_min, g_min, b_min)),"    Normal ",rgbToLuminance((r_max, g_max, b_max)))
    print("TEMP -","Hyper ", rgbToTemperature((r_min, g_min, b_min)),"  Normal ",rgbToTemperature((r_max, g_max, b_max)))
    print("RYB  -","Hyper ", rgbToRyb((r_min, g_min, b_min)),"          Normal ",rgbToRyb((r_max, g_max, b_max)))
    print("XYZ  -","Hyper ", rgbToXyz((r_min, g_min, b_min)),"          Normal ",rgbToXyz((r_max, g_max, b_max)))

    window_size = 3
    center = (randoW[minPxVal],randoH[minPxVal])
    region = cv2.getRectSubPix(img, (window_size, window_size), center)
    resize = cv2.resize(region,(400,400),interpolation=cv2.INTER_CUBIC)
    cv2.imshow("Darker", resize)

    center2 = (randoW[maxPxVal],randoH[maxPxVal])
    region2 = cv2.getRectSubPix(img, (window_size, window_size), center2)
    resize2 = cv2.resize(region2,(400,400),interpolation=cv2.INTER_CUBIC)
    cv2.imshow("Lighter", resize2)

    #glcm_hyper = getGLCMFeatures(gray_img[randoH[minPxVal], randoW[minPxVal]])
    #glcm_normal = getGLCMFeatures(gray_img[randoH[maxPxVal], randoW[maxPxVal]])

    region_hyper = cv2.getRectSubPix(gray_img, (window_size, window_size), center)
    region_normal = cv2.getRectSubPix(gray_img, (window_size, window_size), center2)
    glcm_hyper = getGLCMFeatures(region_hyper)
    glcm_normal = getGLCMFeatures(region_normal)


    print("GLCM Features for Hyper Region:", glcm_hyper)
    print("GLCM Features for Normal Region:", glcm_normal)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

