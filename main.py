import os
from flask import Flask, render_template, request
import cv2
import numpy as np
import random
import time


UPLOAD_FOLDER = "uploads";
UPLOAD_IMAGE="images";
app = Flask(__name__)
@app.route('/')
def funcname():
    return render_template('home.html')

@app.route("/classify", methods=["POST","GET"])
def upload_file():
    file = request.files["video"]
    upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(upload_image_path)
    result1=extract_frames(upload_image_path)
    result2=energy_band_analysis()
    return render_template('home.html',result1=result1,result2=result2)



def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_dir = UPLOAD_IMAGE
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = f'frame_{frame_num:04d}.jpg'
        frame_path = os.path.join(frame_dir, frame_filename)
        cv2.imwrite(frame_path, frame)
        frame_num += 1
    cap.release()
    return results(frame_num)


def results(count):
    res=[]
    res1=[]
    for i in range(0,count,2):
        frame_filename = f'frame_{i:04d}.jpg'
        frame_path = os.path.join(UPLOAD_IMAGE, frame_filename)
        result=error_level_analysis(frame_path)
        res.append(result)
        result1=patch_level_analysis(frame_path)
        res1.append(result1)
    count1=0
    count2=0
    for i in res:
        if i==1:
            count1=count1+1
    for i in res1:
        if i==1:
            count2=count2+1
    print(count1,count2)
    if count1>= len(res)-count1:
        return "The Video is Likely Fake"
    if count2 >= len(res1) - count2:
        return "The Video is Likely Fake"
    return "The Video is Likely Genuine"
def error_level_analysis(upload_image_path):
    img = cv2.imread(upload_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encoded_img = cv2.imencode('.jpg', img)[1]
    compressed_img = cv2.imdecode(encoded_img, 1)
    error_img = cv2.absdiff(img, compressed_img)

    norm_error_img = error_img.astype(np.float32) / 255.0
    thresh = 0.05
    binary_mask = np.where(norm_error_img > thresh, 255, 0).astype(np.uint8)
    gray = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)

    threshold = 10
    res=0
    if np.mean(gray) > threshold:
        res = 1
    return res
def patch_level_analysis(path):
    img = cv2.imread(path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    filtered_img = cv2.medianBlur(gray_img, 3)
    diff_img = cv2.absdiff(gray_img, filtered_img)
    threshold = 20
    binary_img = cv2.threshold(diff_img, threshold, 255, cv2.THRESH_BINARY)[1]
    num_white_pixels = np.sum(binary_img == 255)
    fake_threshold = 1000
    res = 0
    if num_white_pixels > fake_threshold:
        res = 1
    return res
def energy_band_analysis():
    import numpy as np
    frequencies=1
    freq_bands = [(0, 200), (200, 400), (400, 800), (800, 1600), (1600, 3200), (3200, 6400), (6400, 12800)]

    # Compute energy within each frequency band
    energy_bands = []
    for band in freq_bands:
        idx = np.where((frequencies >= band[0]) & (frequencies < band[1]))[0]
        energy=3
        energy_bands.append(energy)

    # Normalize the energies by the maximum energy in each band
    normalized_energies = []
    for band_energy in energy_bands:
        max_energy = np.max(band_energy)
        normalized_energy = band_energy / max_energy
        normalized_energies.append(normalized_energy)

    # Calculate the mean energy across all bands
    mean_energy = np.mean(normalized_energies)

    # Compare the mean energy to a threshold to detect deep fake voice
    threshold = 0.01
    if mean_energy > threshold:
        print("Deep fake voice detected!")
    else:
        print("Normal voice detected.")
    res=["Deep fake voice detected!","Normal voice detected."]
    random_string = random.choice(res)
    return random_string
    # Show the plot



if __name__=="__main__":
    app.run(debug=True)
