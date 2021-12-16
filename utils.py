import PIL
import numpy as np
from PIL import Image


def save_entire_patch_series(input_patches, output_patches):
    for i in range(0, 27):
        Image.fromarray(np.squeeze(input_patches[i]) * 255).convert('L').save(
            '/home/ozlia/prediction2D/input_patch' + str(i) + '.png')
        Image.fromarray(np.squeeze(output_patches[i]) * 255).convert('L').save(
            '/home/ozlia/prediction2D/output_patch' + str(i) + '.png')


def save_img(data_input, data_output, predictions):
    print("saving first image")
    Image.fromarray(np.squeeze(data_input) * 255).convert('L').save('/home/ozlia/prediction2D/input.png')
    Image.fromarray(np.squeeze(predictions) * 255).convert('L').save('/home/ozlia/prediction2D/prediction.png')
    Image.fromarray(np.squeeze(data_output) * 255).convert('L').save('/home/ozlia/prediction2D/orginal.png')

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
