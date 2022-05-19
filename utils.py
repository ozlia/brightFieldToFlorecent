import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import getpass
from patchify import patchify, unpatchify
from sklearn.preprocessing import normalize
from tensorflow import transpose
import imageio
from metrics.metrics import np_corr
from datetime import datetime
from tifffile import imsave as save_tiff

pixel_limit = 65535
USER = getpass.getuser().split("@")[0]
DIRECTORY = "/home/%s" % USER

def set_dir(name):
    global DIRECTORY
    DIRECTORY = "%s/%s" % (DIRECTORY, name)
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)

def get_dir(dir_path):
    return os.path.join(DIRECTORY, dir_path)


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
        norm = cv2.normalize(img[i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        img[i] = (norm/255)
    return img.astype(np.float16)


def save_full_2d_pic(img, name):
    # cv2.imwrite(DIRECTORY + '/' + name, (np.squeeze(img)).astype(np.uint16))
    plt.imsave(DIRECTORY + '/' + name, np.squeeze(img), cmap=plt.cm.gray)


def save_np_as_tiff(img, time, name, model):
    date_dir = DIRECTORY + '/' + model + '_output/' + time
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


def load_numpy_array_v2(name, path):
    p = np.load("/home/%s/%s/%s" % (USER, path, name))
    return p


def save_numpy_array_v2(array, name, path):
    dir_to_save = "/home/%s/%s" % (USER, path)
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)
    np.save(dir_to_save + '/' + name, array)


def transform_dimensions(array, new_shape_indexes):
    return np.array(transpose(array, new_shape_indexes))


def sample_images(model, brightfield_patches, fluorescent_patches, target_dir, fig_name, rescale=False):
    assert len(brightfield_patches.shape) == len(
        fluorescent_patches.shape) == 4, f"Expected 4D array of brightfield and fluorescent images\nArrays shapes received:\n\tBrightfield: {brightfield_patches.shape}\n\tFluorescent: {fluorescent_patches.shape}"

    samples_path = get_dir(os.path.join(target_dir, 'Samples'))
    os.makedirs(samples_path, exist_ok=True)

    rows = brightfield_patches.shape[0]  # 3 is recommended
    cols = 3  # always brightfield,fake fluorescent,fluorescent

    gen_fluorescent = model.predict(brightfield_patches)
    gen_imgs = np.concatenate(
        [brightfield_patches[:, :, :, 0], np.squeeze(gen_fluorescent[:, :, :, 0]), fluorescent_patches[:, :, :, 0]])

    # TODO Rescale images 0 - 1 not sure if necessary
    if rescale:
        gen_imgs = 0.5 * gen_imgs + 0.5

    titles = ['Brightfield', 'Fake Fluorescent', 'Real Fluorescent']
    fig, axs = plt.subplots(rows, cols)
    cnt = 0
    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(gen_imgs[cnt], cmap='gray')
            axs[i, j].set_title(titles[i])
            axs[i, j].axis('off')
            cnt += 1
    fig_path = os.path.join(samples_path, fig_name)
    fig.savefig(fig_path)
    plt.close()


def patchify_predict_imgs(model, imgs, patch_dims):  # assuming img dims are (1,patch_dims)
    patches = utils_patchify(imgs, patch_dims, over_lap_steps=1)
    for row in patches:
        for patch in row:
            patch[0] = model.predict(patch)[0]
    size = imgs[0].shape
    return unpatchify(patches, size)


def reset_dir():
    global DIRECTORY
    DIRECTORY = "/home/%s" % USER #/prediction3D_64px


def calculate_pearson_for_all_images(model, data_input, data_output, time, model_name, organelle):

    print("Numpy corr : -----------")
    all_pearson = []
    for i, img in enumerate(data_input):
        predicted_img = model.predict([img])
        all_pearson.append(np_corr(data_output[i], predicted_img)[0][1])
    all_pearson = np.array(all_pearson)
    file = open("%s/Pearson_correlation_%s.txt" % (DIRECTORY, time), "w+")
    max_index = np.argmax(all_pearson)
    results = "total predicted: %d, mean : %f , std: %f , max value: %f at index: %d" % (len(all_pearson), np.mean(all_pearson) , np.std(all_pearson), np.max(all_pearson), max_index)
    file.writelines([time, "\n", model_name, "\n", organelle, "\n", results, "\n"])
    for i, element in enumerate(all_pearson):
        file.write("%d. score: %f \n" % (i, element))
    file.close()
    print(results)
    print("------------------------------------------------------")
    return max_index

def get_usernames(curr_user_first=True):
    usernames = ['naorsu', 'tomrob', 'ozlia', 'omertag']
    if curr_user_first:
        usernames.insert(usernames.pop(usernames.index(USER)))
    return usernames


def get_time_diff_minutes(first, second):
    assert type(first) == datetime, f'Expected datetime type, received {type(first)}'
    assert type(second) == datetime, f'Expected datetime type, received {type(second)}'
    return ((first - second).total_seconds() // 60) + 1


def save_np_as_tiff_v2(img_channels_last, fname, target_path):
    output_path = get_dir(target_path)
    os.makedirs(output_path, exist_ok=True)

    if img_channels_last.dtype == 'float64':
        img_channels_last.astype(dtype='float32',
                                 casting='same_kind')  # must reduce representation due to ImageJ requirement

    origin_dir = os.getcwd()
    os.chdir(output_path)

    save_tiff(file=f'{fname}.tiff', data=img_channels_last)  # if interesting we have ImageJ param too

    os.chdir(origin_dir)