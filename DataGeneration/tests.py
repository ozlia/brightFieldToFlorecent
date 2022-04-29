from glob import glob
import utils
imgs = glob(utils.get_dir('Mitochondria\\Images\\*.npy'))
print(imgs[0])