import tempfile
import pandas as pd
from DataGeneration.FileStorageTimeComparison.Methods import TimeArrStorage


class Benchmark:
    extension = 'data'
    batch_sizes = [32, 64, 128]
    patch_sizes = [32, 64]  # 128 has very little patches so was rounded up
    img_size = (640, 896)
    assumed_num_imgs_orangelle = 150
    channels = 6

    def __init__(self, method_cls, reps):
        self.num_batches_in_epoch = reps
        self.method_instance: TimeArrStorage = method_cls()
        self.load_times: pd.DataFrame = pd.DataFrame({})

    def run(self):
        '''
        for all possible data_set sizes:
            1. generate data-set based on size
            2. save it to temp folder
            3. time_load rep times
            4. save info

        @return:
        '''

        print('---------------------------------------------')
        print(f'Class {self.method_instance}')

        for patch_size in Benchmark.patch_sizes:
            num_patches = Benchmark.assumed_num_imgs_orangelle * (Benchmark.img_size[0] // patch_size) * (
                    Benchmark.img_size[1] // patch_size)

            # max number of sampling out of an array in NUMPY class is around ~20k because double batch size
            # num_patches = max(num_patches,max(Benchmark.batch_sizes) * self.num_batches_in_epoch * 2 + 1)

            print(f'Saving data for patch size: {patch_size}')
            data_set_size = (num_patches, patch_size, patch_size, Benchmark.channels)  # 24235,32,64,64,6
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                self.method_instance.save(data_set_size, tmp_dir_name)
                for batch_size in Benchmark.batch_sizes:
                    self.method_instance.init_numbers_arr(data_set_size)
                    print(f'\t Timing load for batch size: {batch_size}')
                    for i in range(self.num_batches_in_epoch):
                        self.method_instance.time_load(pth=tmp_dir_name, batch_size=batch_size)
                    self.load_times[
                        f'patch_size_{patch_size}_batch_size_{batch_size}'] = self.method_instance.load_times

                    self.method_instance.load_times.clear()

        self.load_times = self.load_times.add_prefix(str(self.method_instance) + '_')
        print('---------------------------------------------')
