from DataGeneration.FileStorageTimeComparison.Benchmark import Benchmark
from DataGeneration.FileStorageTimeComparison.Methods import METHODS
from numpy.random import seed
import pandas as pd
from os.path import join
from os import makedirs

if __name__ == '__main__':  # Manage benchmarks
    '''
    To test if this works, change:
    1. main: num_batches_in_epoch to 1-2
    2. Benchmark: num_patches to 1-2 (size of file saved) 
    3. Methods: TimeArrStorage.load_idxes to load 1 index at a time  
    '''
    seed(42)
    num_batches_in_epoch = 100
    benchmark_loading_times = []
    makedirs(join('result','raw'),exist_ok=True)

    for method_cls in METHODS:
        curr_benchmark = Benchmark(method_cls=method_cls,reps= num_batches_in_epoch)
        curr_benchmark.run()
        benchmark_loading_times.append(curr_benchmark.load_times)

    benchmark_loading_times = pd.concat(benchmark_loading_times,axis=1)
    for patch_size in Benchmark.patch_sizes:
        cols_by_patch_size = [col for col in benchmark_loading_times.columns if f'patch_size_{patch_size}' in col]
        benchmark_loading_times_by_patch_size = benchmark_loading_times[cols_by_patch_size]
        for batch_size in Benchmark.batch_sizes:
            cols_by_batch_size = [col for col in benchmark_loading_times_by_patch_size.columns if col.endswith(str(batch_size))]
            benchmark_loading_times_by_batch_size = benchmark_loading_times_by_patch_size[cols_by_batch_size]
            benchmark_loading_times_by_batch_size.columns = [col.split('_')[0] for col in benchmark_loading_times_by_batch_size.columns]
            fname = f'patch_size_{patch_size}_batch_size_{batch_size}.csv'
            benchmark_loading_times_by_batch_size.to_csv(join('result','raw',fname))
            benchmark_loading_times_by_batch_size.describe().transpose().to_csv(join('result',fname))

    # fig, ax = plot_results(insts, fname='bm_{0:s}.png'.format(name),
    #                        suptitle='{1:s} storage performance ({2:d}x{3:d}, avg of {0:d}x)'.format(reps, label,
    #                                                                                                 *data.shape))
    # show()
