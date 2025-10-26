import os
import os.path as osp
import numpy as np
import pickle
import logging

root_path = './'
raw_data_file = osp.join(root_path, 'raw_data_test', 'raw_skes_data.pkl')
save_path = osp.join(root_path, 'denoised_data_test')

if not osp.exists(save_path):
    os.mkdir(save_path)

actors_info_dir = osp.join(save_path, 'actors_info')
if not osp.exists(actors_info_dir):
    os.mkdir(actors_info_dir)

missing_count = 0
labels = []

# Set up loggers for various denoising processes
noise_len_logger = logging.getLogger('noise_length')
noise_len_logger.setLevel(logging.INFO)
noise_len_logger.addHandler(logging.FileHandler(osp.join(save_path, 'noise_length.log')))

noise_spr_logger = logging.getLogger('noise_spread')
noise_spr_logger.setLevel(logging.INFO)
noise_spr_logger.addHandler(logging.FileHandler(osp.join(save_path, 'noise_spread.log')))

noise_mot_logger = logging.getLogger('noise_motion')
noise_mot_logger.setLevel(logging.INFO)
noise_mot_logger.addHandler(logging.FileHandler(osp.join(save_path, 'noise_motion.log')))

missing_skes_logger = logging.getLogger('missing_frames')
missing_skes_logger.setLevel(logging.INFO)
missing_skes_logger.addHandler(logging.FileHandler(osp.join(save_path, 'missing_skes.log')))


def get_label_from_name(ske_name):
    """
    Extracts label based on the first letter of ske_name.
    'D' -> label 2, 'N' -> label 1
    """
    # Depression
    if ske_name.startswith('D'):
        return 2
    # Normal
    elif ske_name.startswith('N'):
        return 1
    else:
        raise ValueError(f"Unknown ske_name format: {ske_name}")


def get_one_actor_points(body_data, num_frames):
    """ Get joints for one actor (no color data needed). """
    joints = np.zeros((num_frames, 34), dtype=np.float32)  # 17 joints with x, y coordinates

    start, end = body_data['interval'][0], body_data['interval'][-1]
    joints[start:end + 1] = body_data['joints'].reshape(-1, 34)  # Only x, y

    return joints


def get_raw_denoised_data():
    """ Get denoised joint positions (no color data) from raw skeleton sequences and generate labels. """
    with open(raw_data_file, 'rb') as fr:  # load raw skeletons data
        raw_skes_data = pickle.load(fr)

    num_skes = len(raw_skes_data)
    print(f'Found {num_skes} available skeleton sequences.')

    raw_denoised_joints = []
    frames_cnt = []

    skes_names = []  # List to store ske_name

    for idx, bodies_data in enumerate(raw_skes_data):
        ske_name = bodies_data['name']
        print(f'Processing {ske_name}')

        # Extract label based on ske_name and store it
        label = get_label_from_name(ske_name)
        labels.append(label)

        # only 1 actor
        num_frames = bodies_data['num_frames']
        body_data = list(bodies_data['data'].values())[0]
        joints = get_one_actor_points(body_data, num_frames)

        raw_denoised_joints.append(joints)
        frames_cnt.append(num_frames)

        skes_names.append(ske_name)  # Append ske_name to the list

        if (idx + 1) % 1000 == 0:
            print(f'Processed: {100.0 * (idx + 1) / num_skes:.2f}% ({idx + 1} / {num_skes}), '
                  f'Missing count: {missing_count}')

    # Save the denoised skeleton data
    raw_skes_joints_pkl = osp.join(save_path, 'raw_denoised_joints.pkl')
    with open(raw_skes_joints_pkl, 'wb') as f:
        pickle.dump(raw_denoised_joints, f, pickle.HIGHEST_PROTOCOL)

    # Save frame counts
    frames_cnt = np.array(frames_cnt, dtype=int)
    np.savetxt(osp.join(save_path, 'frames_cnt.txt'), frames_cnt, fmt='%d')

    # Save labels to label.txt
    label_file = osp.join(save_path, 'label.txt')
    np.savetxt(label_file, np.array(labels, dtype=int), fmt='%d')
    print(f'Labels saved to {label_file}')

    dname_file = osp.join(save_path, 'dname.txt')
    with open(dname_file, 'w') as f:
        for ske_name in skes_names:
            f.write(ske_name + '\n')  # Write each ske_name followed by a newline

    print(f'Saved raw denoised positions of {np.sum(frames_cnt)} frames into {raw_skes_joints_pkl}')
    print(f'Found {missing_count} files that have missing data')


if __name__ == '__main__':
    get_raw_denoised_data()
