import cv2
import numpy as np
import math
from scipy.spatial import KDTree
from webcolors import CSS3_HEX_TO_NAMES as css3_hex_to_names
from webcolors import hex_to_rgb
import pandas as pd

COLORS = [
    {"blue": ["aqua", "navy", "teal", "darkblue", "blue", "darkcyan", "darkslateblue", "deepskyblue", "dodgerblue", "lightblue", "lightskyblue", "royalblue"],
     "green": ["green", "lime", "navy", "olive"],
     "gray": ["silver", "slategray", "gray", "darkslategray", "darkgray"],
     "white": ["aliceblue", "azure", "cornsilk", "ghostwhite", "ivory"]
     }
]

def classifyColor_rgb(self, rgb):
    r = rgb[2]
    g = rgb[1]
    b = rgb[0]
        #Training data with color names and their RGB values
    colors = {'red': [255,0,0],
            'green': [0,255,0],
            'blue': [0,0,255],
            'yellow': [255,255,0],
            'cyan': [0,255,255],
            'magenta': [255,0,255],
            'white': [255,255,255],
            'black': [0,0,0],
            'purple': [128,0,128],
            'orange': [255,165,0],
            'pink': [255,192,203],
            'brown': [165,42,42]}

    min_distance = float('inf')
    closest_color = None
    for color, rgb in colors.items():
        # get the smalles distance from obtained color to the set colors
        distance = math.sqrt((rgb[0]-r)**2 + (rgb[1]-g)**2 + (rgb[2]-b)**2)
        if distance < min_distance:
            min_distance = distance
            closest_color = color

        for color in COLORS:
            if closest_color in COLORS[color]:
                return color
            
    return closest_color

def classifyColor_hsv(self, hsv):
    hue = hsv[0]
    saturation = hsv[1]
    value = hsv[2]
    if (value < 40 and saturation <40):  
        return "Black"
    if (value > 210 and saturation >210):  
        return "White"

    if (saturation < 50): 
        return "Gray"

    if (hue < 30):   
        return "Red"
    if (hue < 90):   
        return "Magenta"
    if (hue < 150):  
        return "Blue"
    if (hue < 210):  
        return "Cyan"
    if (hue < 270):  
        return "Green"
    if (hue < 330):  
        return "Yellow"
    return "Red"

def classifyColor_web(rgb_tuple):

    # a dictionary of all the hex and their respective names in css3
    css3_db = css3_hex_to_names
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))
    
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return names[index]

def get_biggest_contour(img):
    R,G,B = cv2.split(img)

    # Do some denosiong on the red chnnale (The red channel gave better result than the gray because it is has more contrast
    Rfilter = cv2.bilateralFilter(R,25,25,10)

    # Threshold image
    ret, Ithres = cv2.threshold(Rfilter,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Find the largest contour and extract it
    contours, contours2 = cv2.findContours(Ithres,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE )

    maxContour = 0
    for contour in contours:
        contourSize = cv2.contourArea(contour)
        if contourSize > maxContour:
            maxContour = contourSize
            maxContourData = contour

    # Create a mask from the largest contour
    mask = np.zeros_like(Ithres)
    cv2.fillPoly(mask,[maxContourData],1)

    # Use mask to crop data from original image
    finalImage = np.zeros_like(img)
    finalImage[:,:,0] = np.multiply(R,mask)
    finalImage[:,:,1] = np.multiply(G,mask)
    finalImage[:,:,2] = np.multiply(B,mask)

    return finalImage

def get_shirt_color(image, shoulder_right, shoulder_left, hip_right, hip_left):
    img_h, img_w, _ = image.shape
    chest_y = (shoulder_left.y + shoulder_right.y) / 2

    # print("getting shirt color")

    if (chest_y) < (hip_right.y):
        # print("chest is higher than hip")
        cut_y_up = int(chest_y * img_h)
        if (hip_right.y) < 1:
            # print("hip is in image")
            cut_y_down = int(hip_right.y * img_h)
        else:
            cut_y_down = int(img_h)
        cut_x_up = int(max(shoulder_right.x, shoulder_left.x) * img_w)
        cut_x_down = int(min(shoulder_right.x, shoulder_left.x) * img_w)

        # margin = 0.1
        # cut_y_up -= int(cut_y_up * margin)
        # cut_y_down += int(cut_y_down * margin)
        # cut_x_up += int(cut_x_up * margin)
        # cut_x_down -= int(cut_x_down * margin)



        # print(f"cut_y_up: {cut_y_up}, cut_y_down: {cut_y_down}, cut_x_up: {cut_x_up}, cut_x_down: {cut_x_down}")
        #cut image from chest to hips
        chestImg = image[cut_y_up:cut_y_down, cut_x_down:cut_x_up]
        #contourImage = self.get_biggest_contour(chestImg)
        
        # cv2.imshow('chestImg', chestImg)
        #cv2.imshow('contourImage', contourImage)

        #get mean color
        mean_color = cv2.mean(chestImg)[:3]
        mean_color = tuple(reversed(mean_color))
        mean_color_rgb = [int(i) for i in mean_color]

        shirtColorweb = classifyColor_web(mean_color_rgb)
        # print('Shirt color (webcolors) is:', shirtColorweb)
        return shirtColorweb
    

# image = cv2.imread('three.jpg')
# shirt_color = get_shirt_color(image)

