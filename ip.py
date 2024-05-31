from flask import Flask, request, send_file
import cv2
import numpy as np
import os

app = Flask(__name__)

def remove_salt_and_pepper_noise(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    median_filtered = cv2.medianBlur(image, 3)
    output_path = 'median_filtered.png'
    cv2.imwrite(output_path, median_filtered)
    return output_path

def remove_gaussian_noise(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((3, 3), np.float32) / 9
    average_filtered = cv2.filter2D(image, -1, kernel)
    output_path = 'average_filtered.png'
    cv2.imwrite(output_path, average_filtered)
    return output_path

@app.route('/process_image', methods=['POST'])
def process_image():
    file = request.files['image']
    noise_type = request.form['noise_type']
    image_path = 'uploaded_image.png'
    file.save(image_path)

    if noise_type == 'salt_and_pepper':
        output_path = remove_salt_and_pepper_noise(image_path)
    elif noise_type == 'gaussian':
        output_path = remove_gaussian_noise(image_path)

    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)






