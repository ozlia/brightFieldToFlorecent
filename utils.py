import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import getpass
from patchify import patchify, unpatchify
from sklearn.preprocessing import normalize
from tensorflow import transpose
import imageio


pixel_limit = 65535
USER = getpass.getuser().split("@")[0]
# print("enter select directory name for this run: ")
DIRECTORY = "/home/%s" % USER


def set_dir(name):
    global DIRECTORY
    DIRECTORY = "%s/%s" % (DIRECTORY, name)
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)

def save_entire_patch_series(input_patches, output_patches):
    global DIRECTORY
    for i in range(0, 27):
        Image.fromarray(np.squeeze(input_patches[i]) * 255).convert('L').save(
            '%s/input_patch_%d.png' % (DIRECTORY, i))
        Image.fromarray(np.squeeze(output_patches[i]) * 255).convert('L').save(
            '%s/output_patch_%d.png' % (DIRECTORY, i))


def save_img(data_input, data_output, predictions):
    print("saving first image")
    global DIRECTORY

    Image.fromarray(np.squeeze(data_input) * 255).convert('L').save('%s/input.png' % DIRECTORY)
    Image.fromarray(np.squeeze(predictions) * 255).convert('L').save('%s/prediction.png' % DIRECTORY)
    Image.fromarray(np.squeeze(data_output) * 255).convert('L').save('%s/original.png' % DIRECTORY)


# todo needs work on defining metrics
def evaluate(test_data_input, test_data_output, metrics):
    """
    prints evaluation
    @param metrics: chosen evaluation metric
    @param test_data_input: list of patches in bright_field
    @param test_data_output: list of patches in fluorescent
    @return: prints metrics results
    """
    # # # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    test_results = []
    for i in range(len(test_data_input)):
        test_results.append(metrics(test_data_input[i], test_data_output[i]))
    print("test evaluation results: ", np.average(test_results))


def norm_img(img):
    # return img / img.max()
    for i in range(img.shape[0]):
        img[i] = (img[i] - np.min(img[i])) / (np.max(img[i]) - np.min(img[i]))
    return img


def save_full_2d_pic(img, name):
    # cv2.imwrite(DIRECTORY + '/' + name, (np.squeeze(img)).astype(np.uint16))
    plt.imsave(DIRECTORY + '/' + name, np.squeeze(img), cmap=plt.cm.gray)


def save_np_as_tiff(img, time, name):
    date_dir = DIRECTORY + '/' + time
    if not os.path.exists(date_dir):
        os.makedirs(date_dir)
    for i in range(img.shape[2]):
        img_slice = img[:, :, i]
        imageio.imwrite("%s/%s_slice-%d_%s.tiff" % (date_dir, name, i, time), img_slice)
        print('.')


def utils_patchify(img_lst, size, resize=False, over_lap_steps=1):
    x, y, z = size
    step = x
    if resize:
        step = int(step / over_lap_steps)
    all_patches = []
    for img in img_lst:
        img_patches = patchify(img, size, step=step)  # split image into 35  128*128 patches. (4, 7, 6, 128, 128, 1)
        if resize:
            all_patches.extend(resize_patch_list(img_patches))
        else:
            all_patches.append(img_patches)
    all_patches = np.array(all_patches)
    if resize: return all_patches
    return all_patches[0]


def resize_patch_list(patches):  # return shape of (28, 128,128,1)
    patches_list4D = []
    for i in np.squeeze(patches):
        for j in i:
            patches_list4D.append(j)
    return patches_list4D


def load_numpy_array(path):
    p = np.load(DIRECTORY + '/' + path)
    return p


def save_numpy_array(array, path):
    np.save(DIRECTORY + '/' + path, array)


def transform_dimensions(array, new_shape_indexes):
    return np.array(transpose(array, new_shape_indexes))
