import os
import os.path as osp
import pickle

import numpy as np

root_path = './'
denoised_path = osp.join(root_path, 'denoised_data_test_3d')
raw_skes_joints_pkl = osp.join(denoised_path, 'raw_denoised_joints.pkl')
frames_file = osp.join(denoised_path, 'frames_cnt.txt')
label_file = osp.join(denoised_path, 'label.txt')  # Labels file generated previously

save_path = './processed_data_test_3d'

if not osp.exists(save_path):
    os.mkdir(save_path)

def align_frames(skes_joints, frames_cnt):
    """ Align sequences with the same frame length """
    num_skes = len(skes_joints)
    max_num_frames = frames_cnt.max()  # Max frame length
    aligned_skes_joints = np.zeros((num_skes, max_num_frames, 51), dtype=np.float32)  # 51 = 17 joints * (x, y, z)

    for idx, ske_joints in enumerate(skes_joints):
        num_frames = ske_joints.shape[0]
        aligned_skes_joints[idx, :num_frames] = ske_joints  # Align the joint data

    return aligned_skes_joints

def split_dataset(skes_joints, labels, save_path):
    # Make sure the path exists
    if not osp.exists(save_path):
        os.makedirs(save_path)

    # Extract data and labels from training or test sets
    test_x = skes_joints
    test_y = one_hot_vector(labels)

    # Save the segmented dataset
    save_name = 'd_gait_test_3d.npz'
    np.savez(osp.join(save_path, save_name), x_test=test_x, y_test=test_y)
    print(f"保存划分数据集到 {save_name}")

def one_hot_vector(labels):
    """ Convert labels to one-hot encoded vectors """
    num_skes = len(labels)
    labels_vector = np.zeros((num_skes, 2))  # Assuming 2 classes (D, N)

    for idx, l in enumerate(labels):
        labels_vector[idx, l - 1] = 1  # Label should be 1 or 2, so adjust by subtracting 1

    # [1, 0] mean label 1 Normal
    # [0, 1] mean label 2 Depression
    return labels_vector

if __name__ == '__main__':
    # Load labels from the label.txt file
    labels = np.loadtxt(label_file, dtype=int)

    # Load frame counts and skeleton joints data
    frames_cnt = np.loadtxt(frames_file, dtype=int)
    with open(raw_skes_joints_pkl, 'rb') as fr:
        skes_joints = pickle.load(fr)  # Load denoised joints data

    # Perform sequence alignment
    skes_joints = align_frames(skes_joints, frames_cnt)

    # Perform random train-test split and save the dataset
    split_dataset(skes_joints, labels, save_path)
