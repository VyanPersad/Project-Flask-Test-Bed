import cv2
import numpy as np
import skimage
from skimage.feature import graycomatrix
from skimage.feature import graycoprops

def extract_glcm_features(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #Calculate GLCM with specified parameters
    distances = [1]  # Distance between pixels
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Angles for pixel pairs
    levels = 256  # Number of gray levels
    symmetric = True
    normed = True
    
    glcm = graycomatrix(gray_image, distances, angles, levels=levels, symmetric=symmetric, normed=normed)
    
    #Calculate GLCM properties
    contrast = graycoprops(glcm, 'contrast').ravel()[0]
    dissimilarity = graycoprops(glcm, 'dissimilarity').ravel()[0]
    homogeneity = graycoprops(glcm, 'homogeneity').ravel()[0]
    energy = graycoprops(glcm, 'energy').ravel()[0]
    correlation = graycoprops(glcm, 'correlation').ravel()[0]
    #entropy = graycoprops(glcm, 'entropy').ravel()[0]
    asm = graycoprops(glcm, 'ASM').ravel()[0]
    #idm = graycoprops(glcm, 'idm').ravel()[0]
    #imc1 = graycoprops(glcm, 'IMC1').ravel()[0]
    #imc2 = graycoprops(glcm, 'IMC2').ravel()[0]
    #max_corr_coeff = graycoprops(glcm, 'maxcorr').ravel()[0]
    #autocorr = graycoprops(glcm, 'autocorr').ravel()[0]
    
    # Print the extracted features
    print("GLCM Contrast:", contrast)
    print("GLCM Dissimilarity:", dissimilarity)
    print("GLCM Homogeneity:", homogeneity)
    print("GLCM Energy:", energy)
    print("GLCM Correlation:", correlation)
    #print("GLCM Entropy:", entropy)
    print("GLCM ASM:", asm)
    #print("GLCM IDM:", idm)
    #print("GLCM IMC1:", imc1)
    #print("GLCM IMC2:", imc2)
    #print("GLCM Maximal Correlation Coefficient:", max_corr_coeff)
    #print("GLCM Autocorrelation:", autocorr)
    
    #return contrast, dissimilarity, homogeneity, energy, correlation, entropy, asm, idm, imc1, imc2, max_corr_coeff, autocorr
    return contrast, dissimilarity, homogeneity, energy, correlation, asm, 

# Load the largest detected skin region image
skin_image = cv2.imread('skin1.png')

# Extract GLCM features from the image
features = extract_glcm_features(skin_image)