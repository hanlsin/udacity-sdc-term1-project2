import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Show feature images.
def show_images(imgs, labels=None, cmap=None, col_cnt=9, row_cnt=5):
    gs = gridspec.GridSpec(row_cnt, col_cnt, hspace=0.5)

    for i in range(np.minimum(row_cnt * col_cnt, len(imgs))):
        ax = plt.subplot(gs[i])

        if cmap is None or len(cmap) == 0:
            ax.imshow(imgs[i])
        else:
            if cmap[i] is None or cmap[i] == "":
                ax.imshow(imgs[i])
            else:
                ax.imshow(imgs[i], cmap=cmap[i])

        if labels is None or len(labels) == 0:
            ax.set_title('index: ' + str(i))
        else:
            ax.set_title(str(labels[i]))

    # Maximize figure
    fig_mng = plt.get_current_fig_manager()
    # for 'wxAgg'
    #fig_mng.frame.Maximize(True)
    # for 'Qt4Agg'
    #fig_mng.window.showMaximized()
    # for 'TkAgg'
    #   for Windows
    fig_mng.resize(*fig_mng.window.maxsize())
    #   for Ubuntu
    #fig_mng.window.state('zoomed')

# Load pickled data
"""
The pickled data is a dictionary with 4 key/value pairs:
* 'features' is a 4D array containing raw pixel data of
    the traffic sign images, (num examples, width, height, channels).
* 'labels' is a 1D array containing the label/class id of the traffic sign.
    The file signnames.csv contains id -> name mappings for each id.
* 'sizes' is a list containing tuples, (width, height) representing the the
    original width and height the image.
* 'coords' is a list containing tuples, (x1, y1, x2, y2) representing
    coordinates of a bounding box around the sign in the image.
    THESE COORDINATES ASSUME THE ORIGINAL IMAGE.
    THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES.
"""
training_file = "./examples/train.p"
validation_file = "./examples/valid.p"
testing_file = "./examples/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# Number of training examples.
n_train = len(X_train)
print("Number of Training examples: ", n_train)

# Number of testing examples.
n_test = len(X_test)
print("Number of Testing examples: ", n_test)

# The shape of an traffic sign image.
image_shape = X_train[0].shape
print("Shape of an traffic sign image: ", image_shape)

# Number of classes.
classes = list(set(y_train))
print("Number of classes in train: ", len(classes))
print("Number of classes in test: ", len(set(y_test)))
print("Number of classes in valid: ", len(set(y_valid)))

"""
# Histogram with the number of each class in the train, test, and valid set.
plt.figure(0)
ax = plt.subplot()
y_multi = [y_train, y_valid, y_test]
ax.hist(y_multi, bins=len(set(y_train)), histtype='bar',
        label=['train', 'valid', 'test'])
ax.legend(prop={'size': 10})
ax.set_title('number of each class')

# Show images of each class.
plt.figure(1)
x_tmp, y_tmp = [], []
for c in classes:
    idx = np.argwhere(y_train == c)[0][0]
    x_tmp.append(X_train[idx])
    y_tmp.append(y_train[idx])
show_images(x_tmp, y_tmp)

#plt.show()
"""
