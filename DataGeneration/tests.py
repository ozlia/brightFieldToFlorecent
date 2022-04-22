from DataGeneration.DataGenPreparation import DataGeneratorPreparation

# img_size = (6, 640, 896)
img_size_channels_last = (640, 896, 6)
patch_size_channels_last = (64, 64, 6)
org_type = 'Mitochondria'
batch_size = 32

dgp = DataGeneratorPreparation(img_size_channels_last=img_size_channels_last, org_type=org_type,patch_size_channels_last=patch_size_channels_last, batch_size=batch_size, resplit=False,
                               validation_size=0.0, test_size=0.1)

dgp.prep_dirs()
dgp.save_images()