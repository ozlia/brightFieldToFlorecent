import PIL
import numpy as np
from PIL import Image
import os
import getpass
from patchify import patchify, unpatchify
import cv2
from aicsimageio import AICSImage

pixel_limit = 65535
USER = getpass.getuser().split("@")[0]
DIRECTORY = "/home/%s/prediction3D" % USER
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
        results = metrics.evaluate(test_data_input, test_data_output)
        print("test loss, test acc:", results)


def norm_img(img):
    return img / pixel_limit


def save_full_2d_pic(img, name):
    cv2.imwrite(DIRECTORY + '/' + name, (np.squeeze(img) * pixel_limit).astype(np.uint16))


def utils_patchify(img_lst, size, resize=False):
    z, y, x = size
    all_patches = []
    for img in img_lst:
        img_patches = patchify(img, size, step=x)  # split image into 35  128*128 patches. (4, 7, 1, 128, 128, 1)
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
