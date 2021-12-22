import PIL
import numpy as np
from PIL import Image
import os
import getpass

USER = getpass.getuser().split("@")[0]
DIRECTORY = "/home/%s/prediction3D" % USER

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
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)
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
