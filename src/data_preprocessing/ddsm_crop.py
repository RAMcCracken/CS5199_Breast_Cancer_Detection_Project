import matplotlib.pyplot as plt
import cv2
import numpy as np
import imutils
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import data_preprocessing.shared_preprocessing as sp

def main():
    #example image
    image_path = '../../img_Calc-Training_P_00008_LEFT_CC.png'
    image = cv2.imread(image_path)
    
    is_left = check_is_left(image_path)
    
    cropped = segment_image(image, is_left)
    filtered = sp.apply_wiener(cropped)
    enhanced = sp.apply_clahe(filtered)
    show_side(image, enhanced)

def check_is_left(image_path):
    if image_path.endswith("LEFT_CC.png") or image_path.endswith("LEFT_MLO.png"):
        is_left = True
    elif image_path.endswith("RIGHT_CC.png") or image_path.endswith("RIGHT_MLO.png"):
        is_left = False
    else:
        print("could not crop image since left or right was not known")
    return is_left

def view_ddsm_image(image):
    # Show sample images
    plt.imshow(image, cmap='gray')
    plt.show()
    # plt.axis('off')
    # plt.savefig("border_coloured_crop")

def show_side(original, new_image):    
    f, (plot1, plot2) = plt.subplots(1, 2)
    plot1.imshow(original)
    plot2.imshow(new_image, cmap="gray")
    plt.show()

def segment_image(image, is_left):
    # load the image and perform pyramid mean shift filtering
    # to aid the thresholding step
    img = image.copy()
    # crop to remove image border
    border_y = img.shape[0]//16
    border_x = img.shape[1]//16
    y = border_y
    h = img.shape[0] - border_y
    #FLIP for RIGHT image
    if is_left:
        x = 10 #start 10 pixels from edge to avoid contours following image boundary
        w = img.shape[1] - (border_x)
    else:
        x = border_x
        w = img.shape[1] - 10 #start 10 pixels from edge to avoid contours following image boundary
    
    img = img[y:h, x:w]
    
    shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)
    
    blur = cv2.blur(shifted,(5,5))
    
    # convert the mean shift image to grayscale, then apply
    # Otsu's thresholding
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts)
    
    largest_contour = cnts[0]
    
    for c in cnts:
        if cv2.contourArea(c) > cv2.contourArea(largest_contour):
            largest_contour = c

    orig = img.copy()
    crop = img.copy()
    x,y,w,h = cv2.boundingRect(largest_contour)
    
    cv2.rectangle(orig,(x,y),(x+w,y+h),(255,0,0),20)

    border_y = image.shape[0]//25
    border_x = image.shape[1]//25
    
    # Crop to bounding box of breast area plus small border
    
    if y - border_y < 0:
        Y1 = y
    else:
        Y1 = y - border_y
    
    if y+h+border_y > image.shape[0]:
        Y2 = y+h
    else:
        Y2 = y+h+border_y
        
    if x + w + border_x:
        X = x + w + border_x
    else:
        X = x + w
        
    crop = crop[Y1:Y2, x:X]

    ((X, Y), _) = cv2.minEnclosingCircle(largest_contour)
    cv2.drawContours(orig, [largest_contour], -1, (0, 255, 0), 20)
    view_ddsm_image(orig)

#   show the output image
    # view_ddsm_image(orig)
    # view_ddsm_image(crop)
    return crop
    
    
if __name__ == '__main__':
    main()