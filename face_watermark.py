import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
from glob import glob


def rotate_image(img, angle):
    """
    Rotate image by angle degrees without cutting corners.
    """
    if angle % 360 == 0:
        return img
    
    size_reverse = np.array(img.shape[1::-1]) # swap x with y
    M = cv2.getRotationMatrix2D(tuple(size_reverse / 2.), angle, 1.)
    MM = np.absolute(M[:,:2])
    size_new = MM @ size_reverse
    M[:,-1] += (size_new - size_reverse) / 2.
    return cv2.warpAffine(img, M, tuple(size_new.astype(int)))


def make_grid(image, x_repeat, y_repeat):
    """
    Create a new image by repeating the input image in a grid.
    """
    aux = np.concatenate([image for i in range(x_repeat)], axis = 1)
    aux = np.concatenate([aux for i in range(y_repeat)], axis = 0)
    return aux


def get_water_mark(image_shape: np.ndarray, watermark_text:str, font_size:int, occurrs: tuple, tilt_angle: float)-> np.ndarray:
    """
    Retun watermark image. It does not apply the watermark to the original image because, in case it's used for a video, this water_mark can be a used multiple times without having to create it again for each frame.
    Params:
        image: image to be watermarked
        watermark_text: text to be used as watermark
        font_size: size of the font
        occurrs: tuple with the number of times the watermark should be repeated in the x and y direction
        tilt_angle: angle of the watermark
    """

    h_image, w_image, *_ = image_shape
    font = ImageFont.truetype("./COOPBL.TTF", font_size)


    # Create a blank image with the same dimensions as the original image
    image_text = Image.new('L', (image_shape[0],image_shape[1]), 0)

    # Get a drawing context
    draw = ImageDraw.Draw(image_text)

    # Draw text
    draw.text((0, 0), watermark_text, font=font, fill=255)
    
    # Get the dimensions of the text image
    left,top,w,h = draw.textbbox(xy= (0,0), text = watermark_text, font=font, spacing=0)

    # Rotate the text image, convert PIL image to numpy array and crop the image
    image_text_np = rotate_image(np.array(image_text)[top:h, :w], angle = tilt_angle)

    # Get the dimensions of the text image
    h_w_single_text = np.array(image_text_np.shape)
    h_single_text, w_single_text = h_w_single_text
        

    if occurrs:
        x_repeat, y_repeat = occurrs
        x_margin_px = int((w_image/x_repeat - w_single_text)/2)
        x_margin = np.zeros((h_single_text, x_margin_px))
        image_text_np = np.concatenate((x_margin, image_text_np, x_margin), axis = 1)

        h_single_text, w_single_text = image_text_np.shape
        y_margin_px = int((h_image/y_repeat - h_single_text)/2)
        y_margin = np.zeros((y_margin_px, w_single_text))
        image_text_np = np.concatenate((y_margin, image_text_np, y_margin), axis = 0)

        water_mark = make_grid(image_text_np, int(x_repeat), int(y_repeat))
        # Convert to uint8
        water_mark = water_mark.astype(np.uint8)

        y_diff = image_shape[0] - water_mark.shape[0]
        x_diff = image_shape[1] - water_mark.shape[1]


        if y_diff:
            top_margin = np.zeros((y_diff//2, water_mark.shape[1]), dtype = np.uint8)
            bottom_margin = np.zeros((y_diff - y_diff//2, water_mark.shape[1]), dtype = np.uint8)
            water_mark = np.concatenate((top_margin, water_mark, bottom_margin), axis = 0)

        if x_diff:
            left_margin = np.zeros((water_mark.shape[0], x_diff//2), dtype = np.uint8)
            right_margin = np.zeros((water_mark.shape[0], x_diff - x_diff//2), dtype = np.uint8)
            water_mark = np.concatenate((left_margin, water_mark, right_margin), axis = 1, dtype = np.uint8)

    else:
        y_repeat, x_repeat = np.ceil(np.array((image_shape[0], image_shape[1]))/h_w_single_text)
        water_mark = make_grid(image_text_np, int(x_repeat), int(y_repeat))[:h_image, :w_image]

    water_mark = np.dstack([water_mark for i in range(3)])

    return water_mark

    
def add_water_mark(image: np.ndarray, watermark: np.ndarray, alpha )-> np.ndarray:
    """
    Wrapper for cv2.addWeighted

    Params:
        image: image to be watermarked
        watermark: watermark image
        alpha: transparency of the watermark
    """
    result_image = cv2.addWeighted(image, 1, watermark, alpha, 0) 
    return result_image


# Examples of use
if __name__ == "__main__":

    ##################################### EXAMPLE 1 #####################################
    # In this examples we will use only one image and vary the parameters

    # Read in the image
    image_path = "./Images/1.jpg"

    image = cv2.imread(image_path)

    watermark_text = 'watermark'

    for font_size in [60, 30]:
        for occurrs in [(3, 2), (2, 3), (1,1)]:
            for transparency in [0.2]:
                for tilt_angle in [30, 45]:
                    # Get watermark
                    water_mark = get_water_mark(image.shape, watermark_text, font_size, occurrs, tilt_angle)

                    # Add watermark to image
                    result_image = add_water_mark(image, water_mark, transparency)

                    # Get basename
                    basename = os.path.basename(image_path)

                    # Save the result
                    path = os.path.join('./Results', f"1_{font_size}_{occurrs[0]}_{occurrs[1]}_{transparency}_{tilt_angle}.jpg")

                    cv2.imwrite(path, result_image)

    #####################################################################################



    ##################################### EXAMPLE 2 #####################################
    # In this example we will use multiple images and fix the parameters
    images = glob('images/*.jpg')
    font_size = 50
    occurrs = (3, 2)
    transparency = 0.2
    tilt_angle = 30

    # Get watermark. Because all the images have the same shape, we can use the same watermark for all of them
    water_mark = get_water_mark((1080, 1920), watermark_text, font_size, occurrs, tilt_angle)

    for image_path in images:
        # Get basename
        basename = os.path.basename(image_path)

        # Path to save the result
        path = os.path.join('./Results', f"{basename}_{font_size}_{occurrs[0]}_{occurrs[1]}_{transparency}_{tilt_angle}.jpg")

        # Read in the image
        image = cv2.imread(image_path)

        # Add watermark to image
        result_image = add_water_mark(image, water_mark, transparency)

        # Save the result
        cv2.imwrite(path, result_image)
        #####################################################################################