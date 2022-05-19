import numpy as np
import utils
from DataGeneration.DataGenPreparation.BasicDataGenPreparation import BasicDataGeneratorPreparation
from DataGeneration.DataGenerator.TestDataGen import TestDataGenerator
from DataGeneration.DataGenerator.TrainDataGen import TrainDataGenerator
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import backend as KB
from run import create_model

# interpreter_path = /home/<username>/.conda/envs/<env name>/bin/python - change your user !!
# interpreter_path_omer  = /home/omertag/.conda/envs/my_env/bin/python

METADATA_CSV_PATH = "/sise/assafzar-group/assafzar/fovs/metadata.csv"
PATCH_SIZE = (3, 64, 64)  # (x,y,z)
FULL_IMG_SIZE = (3, 640, 896)


def run_with_data_gen(dir, model_name, epochs=200, batch_size=32, read_img=False, org_type=None, img_read_limit=150,
        load_model_date=None, over_lap=1, multiply_img_z=1):
    utils.set_dir(dir)
    img_size_rev = (PATCH_SIZE[1], PATCH_SIZE[2], PATCH_SIZE[0])
    start = datetime.now()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("reading images")

    validation_size = 0.0
    test_size = 0.15
    first_time_testing_if_works = False
    num_patches_in_img = (FULL_IMG_SIZE[1] // img_size_rev[0]) * (FULL_IMG_SIZE[2] // img_size_rev[1])

    dgp = BasicDataGeneratorPreparation(img_size_channels_first=FULL_IMG_SIZE,
                                        patch_size_channels_last=img_size_rev, org_type=org_type,
                                        resplit=False, validation_size=validation_size,
                                        test_size=test_size, initial_testing=first_time_testing_if_works)
    print('done saving, starting to allocate data gens')
    train_data_gen = TrainDataGenerator(meta_data_fpath=dgp.images_mapping_fpath,
                                        data_root_path=utils.get_dir(org_type), num_epochs=epochs,
                                        batch_size=batch_size, num_patches_in_img=num_patches_in_img)
    test_data_gen = TestDataGenerator(meta_data_fpath=dgp.images_mapping_fpath,
                                      data_root_path=utils.get_dir(org_type), num_epochs=epochs,
                                      batch_size=batch_size, num_patches_in_img=num_patches_in_img)

    stop = datetime.now()
    print('Done Reading and Patching, Time: ', stop - start)

    # Free up RAM in case the model definition cells were run multiple times
    KB.clear_session()

    print("init model")
    model = create_model(model_name, patch_size_rev=img_size_rev, epochs=epochs, batch_size=batch_size)

    if not load_model_date:
        print("training model")
        model.train(train_x=train_data_gen, train_label=None, val_set=0.1, model_dir=dir)
        stop = datetime.now()
        print('Done Train, Time: ', stop - start)

    else:
        print("Loading model .....")
        model.load_model(model_dir=load_model_date)
        stop = datetime.now()
        print('Done Load, Time: ', stop - start)

    save_time = datetime.now().strftime("%H-%M_%d-%m-%Y")

    br_path = test_data_gen.brightfield_imgs_paths
    fl_path = test_data_gen.fluorescent_imgs_paths
    test_x = []
    test_y = []
    for br_img, fl_img in zip(br_path, fl_path):
        test_x.append(np.load(br_img))
        test_y.append(np.load(fl_img))
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    utils.calculate_pearson_for_all_images(model, test_x[:100], test_y[:100],
                                           model_name=model_name, time=save_time, organelle=org_type)

    print("Generate new pic")
    predicted_img = model.predict([test_x[0]])
    predicted_img_smooth = model.predict_smooth([test_x[0]])  # only if you implanted smooth predict
    print("Saving .........")
    utils.save_np_as_tiff(predicted_img, save_time, "predict", model_name)
    utils.save_np_as_tiff(predicted_img_smooth, save_time, "predict_smooth",
                          model_name)  # only if you implanted smooth predict
    utils.save_np_as_tiff(test_x[0], save_time, "input", model_name)
    utils.save_np_as_tiff(test_y[0], save_time, "ground_truth", model_name)
    print("... All tiffs saved !!")
    stop = datetime.now()
    print('Done All, Time: ', stop - start)
    utils.reset_dir()
