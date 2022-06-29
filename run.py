from helpers import data_prepare, utils
from sklearn.model_selection import train_test_split
from models.CrossDomainAE.crossDomainAE import AutoEncoderCrossDomain
from models.Img2ImgAE.autoEncoder import AutoEncoder
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import backend as KB
import pandas as pd
from argparse import ArgumentParser
# from CrossDomainAE.crossDomainAE import AutoEncoderCrossDomain
from models.UNET.Unet import Unet

# interpreter_path = /home/<username>/.conda/envs/<env name>/bin/python - change your user !!

METADATA_CSV_PATH = "/sise/assafzar-group/assafzar/fovs/metadata.csv"
PATCH_SIZE = (3, 64, 64)  # (x,y,z)
FULL_IMG_SIZE = (3, 640, 896)


def run(dir, model_name, epochs=200, batch_size=32, read_img=False, org_type=None, img_read_limit=150,
        load_model_date=None, over_lap=1, multiply_img_z=1):
    utils.set_dir(dir)
    patch_size_rev = (PATCH_SIZE[1], PATCH_SIZE[2], PATCH_SIZE[0])
    start = datetime.now()
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    data_input, data_output = load_data(read_img, org_type, img_read_limit, multiply_img_z)

    train_x, test_x, train_y, test_y = train_test_split(data_input, data_output, test_size=0.2, random_state=3,
                                                        shuffle=True)
    patches_train_x = utils.utils_patchify(train_x, patch_size_rev, resize=True, over_lap_steps=over_lap)
    patches_train_y = utils.utils_patchify(train_y, patch_size_rev, resize=True, over_lap_steps=over_lap)

    stop = datetime.now()
    print('Done Reading and Patching, Time: ', stop - start)

    # Free up RAM in case the model definition cells were run multiple times
    KB.clear_session()
    model = load_or_train_model(model_name, patch_size_rev, epochs, batch_size, load_model_date, patches_train_x,
                                patches_train_y, start)
    stop = datetime.now()
    predict_images(model, test_x, test_y, model_name, org_type)
    print('Done All, Time: ', stop - start)
    utils.reset_dir()


def load_or_train_model(model_name, patch_size_rev, epochs, batch_size, load_model_date, patches_train_x,
                        patches_train_y, start):
    print("init model")
    model = create_model(model_name, patch_size_rev=patch_size_rev, epochs=epochs, batch_size=batch_size)

    if not load_model_date:
        print("training model")
        model.train(patches_train_x, patches_train_y, val_set=0.2, model_dir=dir)
        stop = datetime.now()
        print('Done Train, Time: ', stop - start)

    else:
        print("Loading model .....")
        model.load_model(model_dir=load_model_date)
        stop = datetime.now()
        print('Done Load, Time: ', stop - start)

    return model


def load_data(read_img, org_type, img_read_limit, multiply_img_z):
    print("reading images")
    paths = data_prepare.load_paths(org_type, limit=img_read_limit)
    if read_img and org_type:
        data_input, data_output = data_prepare.separate_data(paths,
                                                             PATCH_SIZE, multiply_img_z=multiply_img_z)
        utils.save_numpy_array_v2(data_input,
                                  "input_after_prepare_%d_images_%d_z_layer" % (img_read_limit, 3 * multiply_img_z),
                                  "%s_data" % org_type)
        utils.save_numpy_array_v2(data_output,
                                  "output_after_prepare_%d_images_%d_z_layer" % (img_read_limit, 3 * multiply_img_z),
                                  "%s_data" % org_type)
        print("Saved successfully numpy array at /home/%s/%s_data" % (utils.USER, org_type))
    else:
        data_input = utils.load_numpy_array_v2(
            "input_after_prepare_%d_images_%d_z_layer.npy" % (img_read_limit, 3 * multiply_img_z), "%s_data" % org_type)
        data_output = utils.load_numpy_array_v2(
            "output_after_prepare_%d_images_%d_z_layer.npy" % (img_read_limit, 3 * multiply_img_z),
            "%s_data" % org_type)
        print("Loaded successfully %d images as numpy array from /home/%s/%s_data" % (
            len(data_output), utils.USER, org_type))

    data_input = utils.transform_dimensions(data_input, [0, 2, 3, 1])
    data_output = utils.transform_dimensions(data_output, [0, 2, 3, 1])
    return data_input, data_output


def predict_images(model, test_x, test_y, model_name, org_type):
    save_time = datetime.now().strftime("%H-%M_%d-%m-%Y")
    max_index = utils.calculate_pearson_for_all_images(model, test_x[:100], test_y[:100],
                                                       model_name=model_name, time=save_time, organelle=org_type)

    print("Generate new pic")
    predicted_img = model.predict([test_x[max_index]])
    predicted_img_smooth = model.predict_smooth([test_x[max_index]])  # only if you implanted smooth predict
    print("Saving .........")
    utils.save_np_as_tiff(predicted_img, save_time, "best_predict", model_name)
    utils.save_np_as_tiff(predicted_img_smooth, save_time, "best_predict_smooth",
                          model_name)  # only if you implanted smooth predict
    utils.save_np_as_tiff(test_x[max_index], save_time, "best_input", model_name)
    utils.save_np_as_tiff(test_y[max_index], save_time, "best_ground_truth", model_name)

    print("Generate 0 pic")
    predicted_img = model.predict([test_x[0]])
    predicted_img_smooth = model.predict_smooth([test_x[0]])  # only if you implanted smooth predict
    print("Saving .........")
    utils.save_np_as_tiff(predicted_img, save_time, "0_predict", model_name)
    utils.save_np_as_tiff(predicted_img_smooth, save_time, "0_predict_smooth",
                          model_name)  # only if you implanted smooth predict
    utils.save_np_as_tiff(test_x[0], save_time, "0_input", model_name)
    utils.save_np_as_tiff(test_y[0], save_time, "0_ground_truth", model_name)

    print("... All tiffs saved !!")


def create_model(name: str, patch_size_rev, epochs, batch_size):
    '''
    name - Must be lower case in this function!!!
    '''

    name = name.lower()

    if name == "img2img":
        return AutoEncoder(patch_size_rev, epochs=epochs, batch_size=batch_size)
    elif name == "b2b":
        return AutoEncoderCrossDomain(patch_size_rev, epochs=epochs, batch_size=batch_size)
    elif name == "f2f":
        return AutoEncoderCrossDomain(patch_size_rev, epochs=epochs, batch_size=batch_size)
    elif name == "unet":
        return Unet(patch_size_rev, epochs=epochs, batch_size=batch_size)
    elif name == "pix2pix":
        return None
    else:
        return None


def parse_command_line():
    parser = ArgumentParser(description="Getting arguments to run model")
    parser.add_argument("-m", "--model_type", nargs=1, required=True, help="Model to run [img2img, crossdomain, unet]")
    parser.add_argument("-d", "--dir", default=["run_%s" % datetime.now().strftime("%d-%m-%Y_%H-%M")],
                        nargs=1, help="Directory name to save")
    parser.add_argument("-e", "--epochs", nargs=1, default=1000, type=int, help="Number of epochs")
    parser.add_argument("-bz", "--batch_size", nargs=1, default=32, type=int, help='Batch size')
    parser.add_argument("-ri", "--read_img", action='store_true', default=False,
                        help='Use this flag to read new images')
    parser.add_argument("-o", "--org_type", nargs=1, default=None, type=str,
                        help='Name of Organelle from list: %s' % ', '.join(
                            sorted(set(pd.read_csv(METADATA_CSV_PATH)['StructureDisplayName']))))
    parser.add_argument("-rl", "--read_limit", default=150, type=int, help='Maximum number of images to read')
    args = parser.parse_args()
    run(dir=args.dir, model_name=args.model_type, epochs=args.epochs, batch_size=args.batch_size,
        read_img=args.read_img, org_type=args.org_type[0], img_read_limit=args.read_limit)


def run_all_orgs(selected_model_name: str, best_orgs_dict: dict,
                 epochs=100, batch_size=32, read_img=True, img_read_limit=150, multiply_img_z=2):
    """
    Params:
    selected_model - name of the model
    --------------
    best_orgs dict -

    Key: Organelle Name.
    Example - "Mitochondria"

    Value: Model date-time to load, to create new model leave None.
    Example - "28-04-2022_10-59"
    """

    start_all = datetime.now()

    for organelle, model_date in best_orgs_dict.items():
        working_org = "----- Working on organelle: %s -----" % organelle
        print(len(working_org) * "-")
        print(working_org)
        print(len(working_org) * "-")

        model_dir = "/%s_%s_%s/" % (selected_model_name, organelle, model_date) if model_date else None

        run(dir="%s_%s" % (selected_model_name, organelle), model_name=selected_model_name, epochs=epochs,
            batch_size=batch_size,
            read_img=read_img, org_type=organelle, img_read_limit=img_read_limit, load_model_date=model_dir,
            multiply_img_z=multiply_img_z)

        done_org = "***** Done organelle: %s *****" % organelle
        print(len(done_org) * "*")
        print(done_org)
        print(len(done_org) * "*")

    stop_all = datetime.now()
    print('All organelles done, Total Time for this run: ', stop_all - start_all)


if __name__ == '__main__':
    # todo please change your run params here
    # see run_all_orgs function and documentation
    # you can copy model name from here

    all_models = ["pix2pix", "unet", "f2f", "b2b", "img2img"]

    selected_model = "unet"
    organelle = "Mitochondria"

    # todo please comment/uncomment your selected Organelle !!
    best_orgs = {
        # "Mitochondria": None, #"18-05-2022_21-48",
        # "Microtubules": None, #"18-05-2022_23-34",
        # "Endoplasmic-reticulum": None, #"19-05-2022_01-59",
        # "Nuclear-envelope": None, #"19-05-2022_04-16",
        # "Actin-filaments": None, #"18-05-2022_05-11"
        # "Tight-junctions": None,
        # "Nucleolus-(Dense-Fibrillar-Component)": None,
        # "Peroxisomes": None, ## Need to find br and fl channel before run!!
        # "Golgi": None,
        # "Endosomes": None, ## Need to find br and fl channel before run!!
        "Gap-junctions": None,
        "Lysosome": None,
        "Adherens junctions": None,
        "Nucleolus-(Granular-Component)": None,
        "Matrix-adhesions": None,
        "Actomyosin-bundles": None,
        "Desmosomes": None,
        "Plasma-membrane": None
    }

    # for org, val in best_orgs.items():
    #     TIFFS_DF = utils.all_tiff_df(organelle_name=org)

    # utils.organelle_list()
    run_all_orgs(selected_model, best_orgs,
                 epochs=100, batch_size=32, read_img=True, img_read_limit=500, multiply_img_z=2)

    # run_with_data_gen(dir="%s_%s" % (selected_model, organelle), model_name=selected_model, epochs=10, batch_size=64,
    #                   org_type=organelle, load_model_date=None, over_lap=1, multiply_img_z=1)

    # organelle = "Mitochondria"
    # run(dir="%s_%s" % (selected_model, organelle), model_name=selected_model, epochs=100, batch_size=32, read_img=True,
    #     org_type=organelle, img_read_limit=200, multiply_img_z=4)
