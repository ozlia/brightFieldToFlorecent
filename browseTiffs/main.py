from aicsimageio import AICSImage
import numpy as np
import matplotlib.pyplot as plt
import os
dir_type = "Mitochondria"
root_dir = os.path.join("/storage/users/assafzar/fovs",dir_type)
fovs = os.listdir(root_dir)

counter = 9
for tiff in fovs[counter:counter+1]:
    curr_tiff_name = os.path.join(root_dir,tiff)
    reader = AICSImage(curr_tiff_name)
    img = reader.data
    # print(img.shape)

    img = np.squeeze(img, axis=0)
    # print(img.shape)

    n_channels = img.shape[0]
    mid_slice = np.int(0.5 * img.shape[1])
    fig, ax = plt.subplots(1, n_channels, figsize=(18, 16), dpi=72)

    for channel in range(n_channels):
        ax[channel].axis('off')
        ax[channel].imshow(img[channel, mid_slice, :, :], cmap=plt.cm.gray)
    plt.title(str(counter) + ") " + tiff)
    plt.show()
    counter = counter + 1

