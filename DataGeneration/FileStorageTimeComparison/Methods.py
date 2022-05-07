from time import time
import numpy as np
import h5py
from numpy import save as np_save, load as np_load, savez
from zarr import load as zarr_load, open as zarr_open
import threading
from os.path import join
from time import time as now

class TimeArrStorage(object):
    extension = 'data'

    def __init__(self):
        self.load_times = []
        self.idxs = None

    def init_numbers_arr(self, arr_dims):
        self.numbers_range = np.arange(start=0, stop=arr_dims[0], step=1)

    def __str__(self):
        return self.__class__.__name__

    def load_idxs(self, arr_dims, batch_size):
        # self.idxs = np.random.choice(a=self.numbers_range, size=2, replace=False)
        self.idxs = np.random.choice(a=self.numbers_range, size=batch_size, replace=False)
        self.numbers_range = np.setdiff1d(self.numbers_range, self.idxs)

    def save(self, arr, pth):
        # implementations have to call `sync`!
        raise NotImplementedError

    def load(self, pth, batch_size) -> int:  # Must call sum of each loaded arr to ensure it isn't lazy-loaded
        raise NotImplementedError

    def time_load(self, pth, arr_dims, batch_size):
        self.load_idxs(arr_dims, batch_size)
        t0 = now()
        self.load(pth)
        self.load_times.append(now() - t0)

    @classmethod
    def generate_arr(cls, dims):
        return np.ones(dims)


class NPY(TimeArrStorage):
    extension = 'npy'

    def load_idxs(self, arr_dims, batch_size):
        super(NPY, self).load_idxs(arr_dims, batch_size=batch_size * 2)

    def save(self, arr_dims, pth):
        all_imgs_in_data = TimeArrStorage.generate_arr(arr_dims)
        for i in range(arr_dims[0]):
            img_pth = self.get_img_name(pth, i)
            with open(img_pth, 'wb+') as fh:
                np_save(fh, all_imgs_in_data[i], allow_pickle=False)

    def load(self, pth):
        for idx in self.idxs:
            img_pth = self.get_img_name(pth, idx)
            arr = np_load(img_pth)
            arr.sum()

    def get_img_name(self, path, idx):
        return f'{path}_{idx}'


class NPZ(TimeArrStorage):
    extension = 'npz'

    def save(self, arr_dims, pth):
        all_imgs_in_data = TimeArrStorage.generate_arr(arr_dims)
        for i in range(arr_dims[0]):
            pair_path = self.get_pair_name(pth, i)
            with open(pair_path, 'wb+') as fh:
                savez(fh, brightfield=all_imgs_in_data[i], fluorescent=all_imgs_in_data[i])

    def load(self, pth):
        for idx in self.idxs:
            img_pth = self.get_pair_name(pth, idx)
            arr = np_load(img_pth)
            arr['brightfield'].sum()
            arr['fluorescent'].sum()

    def get_pair_name(self, path, idx):
        return f'{path}_{idx}'


class HDF5LargeArray(TimeArrStorage):
    extension = 'hdf5'

    def save(self, arr_dims, pth, with_chunks=None):
        all_imgs_in_data = TimeArrStorage.generate_arr(arr_dims)
        fname = self.get_fname(pth)
        with h5py.File(fname, 'a') as fh:
            train_grp = fh.create_group('train')
            train_grp.create_dataset(name='brightfield_patches', data=all_imgs_in_data, chunks=with_chunks)
            train_grp.create_dataset(name='fluorescent_patches', data=all_imgs_in_data, chunks=with_chunks)
            fh.flush()

    def load(self, pth):
        fname = self.get_fname(pth)
        with h5py.File(fname, 'r') as fh:
            brightfield_patches = fh['train']['brightfield_patches'][self.idxs]
            fluorescent_patches = fh['train']['fluorescent_patches'][self.idxs]
            # Do something with the data, as it is lazy-loaded
            _ = brightfield_patches.sum()
            _ = fluorescent_patches.sum()

    def get_fname(self, path):
        return f'{path}_testfile.{HDF5LargeArray.extension}'


class HDF5AutomatedChunkyLargeArray(HDF5LargeArray):
    def save(self, arr_dims, pth, with_chunks=False):
        super(HDF5AutomatedChunkyLargeArray, self).save(arr_dims, pth, with_chunks=True)


class HDF5Pairs(TimeArrStorage):  # FAN FAVORITE
    extension = 'hdf5'

    def save(self, arr_dims, pth, with_chunks=True):  # 20k,2,64,64,6
        arr_dims = (arr_dims[0], 2,) + arr_dims[1:]  # pairs of images
        all_imgs_in_data = TimeArrStorage.generate_arr(arr_dims)
        fname = self.get_fname(pth)
        with h5py.File(fname, 'a') as fh:
            train_grp = fh.create_group('train')
            train_grp.create_dataset(name='patches_pairs', data=all_imgs_in_data, chunks=with_chunks)
            fh.flush()

    def load(self, pth):  # 32,2,64,64,6 -> 2 x 32,64,64,6
        fname = self.get_fname(pth)
        with h5py.File(fname, 'r') as fh:
            pairs_patches = fh['train']['patches_pairs'][self.idxs]
            brighfield_patches, fluorescent_patches = np.array_split(ary=pairs_patches, indices_or_sections=2, axis=1)
            brighfield_patches = np.squeeze(brighfield_patches, axis=1)
            fluorescent_patches = np.squeeze(fluorescent_patches, axis=1)
            # Do something with the data, as it is lazy-loaded
            _ = brighfield_patches.sum()
            _ = fluorescent_patches.sum()

    def get_fname(self, path):
        return f'{path}_testfile.{HDF5LargeArray.extension}'


class ZARR(TimeArrStorage):
    extension = 'zarr'

    def save(self, arr_dims, pth, with_chunks=True):
        all_imgs_in_data = TimeArrStorage.generate_arr(arr_dims)
        fname = self.get_fname(pth)
        fh = zarr_open(fname, mode='w')
        train_grp = fh.create_group('train')
        train_grp.create_dataset(name='brightfield_patches', data=all_imgs_in_data, chunks=with_chunks)
        train_grp.create_dataset(name='fluorescent_patches', data=all_imgs_in_data, chunks=with_chunks)

    def one_process_load(self,pth,idxs,data_format):
        fname = self.get_fname(pth)
        fh = zarr_load(fname)
        patches = fh[join('train',f'{data_format}_patches')][idxs]
        # patches = fh['train'][f'{data_format}_patches']
        _ = patches.sum()

    def load(self, pth):
        brightfield_idxs = self.idxs[:len(self.idxs) // 2]
        fluorescent_idxs = self.idxs[len(self.idxs) // 2:]
        brightfield_reader = threading.Thread(target=self.one_process_load,args=(pth,brightfield_idxs,'brightfield',))
        fluorescent_reader = threading.Thread(target=self.one_process_load,args=(pth,fluorescent_idxs,'fluorescent',))
        brightfield_reader.start()
        fluorescent_reader.start()
        brightfield_reader.join()
        fluorescent_reader.join()

    def get_fname(self, path):
        return f'{path}_testfile.{ZARR.extension}'

METHODS = (
    NPY,
    NPZ,
    HDF5LargeArray,
    HDF5AutomatedChunkyLargeArray,
    HDF5Pairs,
    ZARR,
)
