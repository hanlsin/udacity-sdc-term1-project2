### Preprocess the data here.
# Preprocessing steps could include normalization, converting to grayscale, etc.
import cv2
import matplotlib.pyplot as plt
import numpy as np
from load_data import show_images, image_shape, X_train, y_train, X_test, y_test

# convert image to the gray scale image.
def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# normalize image data with Min-Max scaling.
def normalize(image_data):
    o_min = 0.1
    o_max = 0.9

    gray_min = 0
    gray_max = 255

    # ax + b
    a = (o_max - o_min) / (gray_max - gray_min)
    b = o_min

    return a * image_data + b

# normalize with OpenCV normalize Min-Max.
def cv2_normalize(image_data):
    return cv2.normalize(image_data, image_shape, alpha=0, beta=1,
                         norm_type=cv2.NORM_MINMAX,
                         dtype=cv2.CV_32F)

# preprocess image data.
def preprocess(image_data):
    # convert gray scale image
    gray = []
    for img in image_data:
        gray.append(convert_to_gray(img))

    # normalize gray image data
    norm = cv2_normalize(np.array(gray))

    return gray, norm

print("Image shape: " + str(image_shape))
X_train_gray, X_train_norm = preprocess(X_train)
print(X_train_norm.shape)
X_test_gray, X_test_norm = preprocess(X_test)
print(X_test_norm.shape)

"""
# Show few converted images
c_cnt = 12
r_cnt = 5
x_tmp, cmap_tmp, label_tmp = [], [], []
for i in np.random.random_integers(0, high=len(X_train), size=r_cnt * c_cnt):
    # original image
    org_img = X_train[i]
    x_tmp.append(org_img)
    cmap_tmp.append("")
    label_tmp.append("[" + str(y_train[i]) + "]")
    # denoise image
    x_tmp.append(cv2.fastNlMeansDenoisingColored(X_train[i], None, 10, 3, 7))
    cmap_tmp.append("")
    label_tmp.append("Deno")
    # gray scale image
    x_tmp.append(X_train_gray[i])
    cmap_tmp.append("gray")
    label_tmp.append("+[" + str(i) + "]")
    # normalized image
    x_tmp.append(X_train_norm[i])
    cmap_tmp.append("")
    label_tmp.append("+[" + str(i) + "]")
plt.figure(2)
show_images(x_tmp, labels=label_tmp, cmap=cmap_tmp,
            col_cnt=c_cnt, row_cnt=r_cnt)
plt.show()
"""
