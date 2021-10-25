from Pix2Pix import data_loader
from BasicAE import data_prepere
if __name__ == '__main__':
    org_type = "Mitochondria/"
    # dl = data_loader.DataLoader()
    # b = dl.load_batch(batch_size=32,is_testing=False)
    # print(b)
    single_mit_image = data_prepere.load(org_type)


