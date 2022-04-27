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

pixel_limit = 65535
USER = getpass.getuser().split("@")[0]
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


def transform_dimensions(array, new_shape_indexes):
    return np.array(transpose(array, new_shape_indexes))


def sample_images(model, brightfield_imgs, fluorescent_imgs, fig_name, rescale=False, org_type=None):
    assert len(brightfield_imgs.shape) == len(
        fluorescent_imgs.shape) == 4, f"You must send a 4D array of brightfield and fluorescent images\nArrays shapes received:\n\tBrightfield: {brightfield_imgs.shape}\n\tFluorescent: {fluorescent_imgs.shape}"

    imgs_path = 'sampled_images'
    if org_type is not None:
        imgs_path = os.path.join(org_type, imgs_path)
    images_root_dir = os.path.join(DIRECTORY, imgs_path)
    os.makedirs(images_root_dir, exist_ok=True)

    rows = brightfield_imgs.shape[0]  # 3 is recommended
    cols = 3  # always brightfield,gen fluorescent,fluorescent

    gen_fluorescent = model.predict(brightfield_imgs)
    gen_imgs = np.concatenate(
        [brightfield_imgs[:, :, :, 0], np.squeeze(gen_fluorescent[:, :, :, 0]), fluorescent_imgs[:, :, :, 0]])

    # TODO Rescale images 0 - 1 not sure if necessary
    if rescale:
        gen_imgs = 0.5 * gen_imgs + 0.5

    titles = ['brightfield', 'gen fluorescent', 'real fluorescent']
    fig, axs = plt.subplots(rows, cols)
    cnt = 0
    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(gen_imgs[cnt], cmap='gray')
            axs[i, j].set_title(titles[i])
            axs[i, j].axis('off')
            cnt += 1
    fig_path = os.path.join(images_root_dir, fig_name)
    fig.savefig(fig_path)
    # plt.show()
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
