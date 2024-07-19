import numpy as np
import cv2
import os
import shutil
import argparse


def check_image(image_path, brightness_threshold = 240, undersaturation_threshold = 50):

    status = "ok"

    ################################################################
    # Check overexposition
    ################################################################
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Count the number of pixels above the threshold
    num_saturated_pixels = cv2.countNonZero(cv2.inRange(gray_image, brightness_threshold, 255))

    # Calculate the ratio of saturated pixels to total pixels
    total_pixels = gray_image.size
    saturation_ratio = num_saturated_pixels / total_pixels

    # If the ratio is above a certain threshold, consider the image overexposed
    overexposed_threshold = 0.05  # You can adjust this threshold as needed

    is_overexposed = saturation_ratio > overexposed_threshold

    if is_overexposed:
        status = "overexposed"
    
    ################################################################
    # End Check overexposition
    ################################################################

    
    ################################################################
    # Check blueish
    ################################################################
    
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    blue_lower = np.array([100, 50, 50], dtype=np.uint8)
    blue_upper = np.array([140, 255, 255], dtype=np.uint8)

    # Create masks to filter out dark and blueish regions
    blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)

    # Check if there are dark or blueish pixels in the image
    blue_pixels = cv2.countNonZero(blue_mask)

    num_image_pixels = image.shape[0] * image.shape[1]
    
    is_blueish = blue_pixels > (num_image_pixels // 2)
    if is_blueish:
        if status == "ok":
            status = "blueish"
        else:
            status = status + "_blueish"
    
    ################################################################
    # End Check blueish
    ################################################################


    ################################################################
    # Check darkness
    ################################################################

    # Calculate histogram
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # Calculate total number of pixels
    total_pixels = gray_image.shape[0] * gray_image.shape[1]

    # Calculate cumulative histogram
    cumulative_hist = np.cumsum(hist)

    # Calculate cumulative percentage
    cumulative_percentage = cumulative_hist / total_pixels

    is_dark = np.argmax(cumulative_percentage > 0.8) < 65 #80 percent of pixels have value below 75

    if is_dark:
        if status == "ok":
            status = "dark"
        else:
            status = status + "_dark"
    
    ################################################################
    # End Check darkness
    ################################################################
            
    ################################################################
    # Check undersaturation
    ################################################################
    
     # Calculate mean saturation level
    mean_saturation = np.mean(hsv_image[:,:,1])
    
    # Determine if image is undersaturated
    is_undersaturated = mean_saturation < undersaturation_threshold
    if is_undersaturated:
        if status == "ok":
            status = "undersaturated"
        else:
            status = status + "_undersaturated"
    
    ################################################################
    # End Check undersaturation
    ################################################################
    
    
    return status


def main():

    parser = argparse.ArgumentParser(
        description="Program to filter the generated samples.")
    parser.add_argument("source", type=str, help="source directory path")
    args = parser.parse_args()

    folder_path = args.source
    folder_path = "RESULTS/20240315-194356_sample_2024-03-06_All_Samples_as_PNGs_d=300_c=15_samples=10000_model=ema_0.9999_053460.pt_s=1.5/samples"
    image_paths = [p for p in os.listdir(folder_path) if p.endswith(".png")]
    image_states = [check_image(os.path.join(folder_path, path)) for path in image_paths]
    print(np.unique(image_states, return_counts=True))

    # for path, state in zip(image_paths, image_states):
    # full_path = os.path.join(folder_path, path)
    # if not os.path.isdir(os.path.join(folder_path, os.pardir, state, os.path.basename(folder_path))):
    #     os.makedirs(os.path.join(folder_path, os.pardir, state, os.path.basename(folder_path)))
    # shutil.copyfile(full_path, os.path.join(folder_path, os.pardir, state, os.path.basename(folder_path), path))


if __name__ == "__main__":
    main()



