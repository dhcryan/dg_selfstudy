import pickle
import matplotlib.pyplot as plt
#Sample 하나당 plotting하는 건데 
import sys
import mne
import numpy as np
import os
from multiprocessing import Process
import pickle
import argparse

root = 'C:\\Users\\dhc40\\manyDG\\data\\sleep\\sleep-edf-database-expanded-1.0.0'
def sample_process(root_folder, k, N, epoch_sec, index):
    for i, j in enumerate(index,5):
        if i % N == k:
            if k == 0:
                print ('Progress: {} / {}'.format(i, len(index)))

            root_folder = os.path.join(root, 'sleep-cassette')
            pat_files = list(filter(lambda x: x[:5] == j, os.listdir(root_folder)))
            pat_nights = [item[:6] for item in pat_files]
            for pat_per_night in pat_nights:
                # load signal "X" part
                data = mne.io.read_raw_edf(root_folder + '\\' + list(filter(lambda x: (x[:6] == pat_per_night) and ('PSG' in x), pat_files))[0])
                X = data.get_data()[:, :]
                print(X.shape)
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--windowsize', type=int, default=30, help="unit (seconds)")
    parser.add_argument('--multiprocess', type=int, default=20, help="How many processes to use")
    args = parser.parse_args()

    
    out_root = os.path.join(root, "cassette_processed")
    if not os.path.exists(out_root):
        os.makedirs(out_root)

    data_folder = os.path.join(root, 'sleep-cassette')

    all_index = np.unique([path[:5] for path in os.listdir(data_folder)])
    N, epoch_sec = args.multiprocess, args.windowsize
    p_list = []
    for k in range(N):
        process = Process(target=sample_process, args=(root, k, N, epoch_sec, all_index))
        process.start()
        p_list.append(process)

    # for i in p_list:
    #     i.join() 
# with open('C:\\Users\\dhc40\\manyDG\\data\\sleep\\sleep-edf-database-expanded-1.0.0\\cassette_processed\\cassette-SC4731-1265.pkl', 'rb') as f:
#     data = pickle.load(f)
#     # print(obj['X'].shape)
#     # f.close()
#     # X = data['X']
#     # print(X[0])
# # X, y 추출
# X = data['X']
# y = data['y']

# # plot
# fig, axs = plt.subplots(nrows=2, figsize=(10, 5))
# axs[0].plot(X[0])
# axs[0].set_title('Channel 1')
# axs[1].plot(X[1])
# axs[1].set_title('Channel 2')
# fig.suptitle(f'Sleep stage: {y}')
# plt.show()