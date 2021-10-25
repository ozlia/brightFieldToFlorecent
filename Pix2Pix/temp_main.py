from sklearn.model_selection import train_test_split
from BasicAE import data_prepere


org_type = "Mitochondria/"
dl = data_prepere.load(org_type)
res = train_test_split(dl)
print(len(res))
# b = dl.load_batch(batch_size=32,is_testing=False)
# print(b)
# imgs_paths = data_prepere.load(org_type)
# res = data_prepere.load_images_as_batches(brightfield_fluorescent_tiff_paths=imgs_paths,batch_size=2        ,img_res=(128,128))
# print(res[0].shape)




