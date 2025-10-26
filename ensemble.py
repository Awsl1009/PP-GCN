import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha',
                        default=1,
                        help='weighted summation',
                        type=float)

    parser.add_argument('--joint-dir',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone-dir',
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')
    parser.add_argument('--joint-motion-dir', default=None)
    parser.add_argument('--bone-motion-dir', default=None)
    parser.add_argument('--output-dir', default='./output', help='Directory to save the confusion matrix image')

    arg = parser.parse_args()

    npz_data = np.load('./data/processed_data_test/' + 'd_gait_test.npz')
    label = np.where(npz_data['y_test'] > 0)[1]

    with open(os.path.join(arg.joint_dir, 'epoch1_test_score.pkl'), 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(os.path.join(arg.bone_dir, 'epoch1_test_score.pkl'), 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    if arg.joint_motion_dir is not None:
        with open(os.path.join(arg.joint_motion_dir, 'epoch1_test_score.pkl'), 'rb') as r3:
            r3 = list(pickle.load(r3).items())
    if arg.bone_motion_dir is not None:
        with open(os.path.join(arg.bone_motion_dir, 'epoch1_test_score.pkl'), 'rb') as r4:
            r4 = list(pickle.load(r4).items())

    all_labels = []
    all_preds = []

    if arg.joint_motion_dir is not None and arg.bone_motion_dir is not None:
        arg.alpha = [0.8, 1.0, 0.2, 0.2]
        for i in tqdm(range(len(label))):
            l = label[i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            _, r33 = r3[i]
            _, r44 = r4[i]
            r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2] + r44 * arg.alpha[3]

            # Predict the label with the highest score
            pred = np.argmax(r)
            all_labels.append(int(l))
            all_preds.append(pred)

        # Compute Precision, Recall, and F1
        precision = precision_score(all_labels, all_preds, zero_division=0, average='binary')
        recall = recall_score(all_labels, all_preds, zero_division=0, average='binary')
        f1 = f1_score(all_labels, all_preds, zero_division=0, average='binary')

    elif arg.joint_motion_dir is not None and arg.bone_motion_dir is None:
        arg.alpha = [0.6, 0.6, 0.4]
        for i in tqdm(range(len(label))):
            l = label[i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            _, r33 = r3[i]
            r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2]

            # Predict the label with the highest score
            pred = np.argmax(r)
            all_labels.append(int(l))
            all_preds.append(pred)

        # Compute Precision, Recall, and F1
        precision = precision_score(all_labels, all_preds, zero_division=0, average='binary')
        recall = recall_score(all_labels, all_preds, zero_division=0, average='binary')
        f1 = f1_score(all_labels, all_preds, zero_division=0, average='binary')

    else:
        for i in tqdm(range(len(label))):
            l = label[i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            r = r11 + r22 * arg.alpha

            # Predict the label with the highest score
            pred = np.argmax(r)
            all_labels.append(int(l))
            all_preds.append(pred)

        # Compute Precision, Recall, and F1
        precision = precision_score(all_labels, all_preds, zero_division=0, average='binary')
        recall = recall_score(all_labels, all_preds, zero_division=0, average='binary')
        f1 = f1_score(all_labels, all_preds, zero_division=0, average='binary')

    print(f'Precision: {precision * 100:.4f}%')
    print(f'Recall: {recall * 100:.4f}%')
    print(f'F1 Score: {f1 * 100:.4f}%')

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Normalize the confusion matrix to display percentages
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Plot confusion matrix heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Depressed'], yticklabels=['Normal', 'Depressed'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Depression Detection')

    # Save the confusion matrix heatmap as a file (e.g., PNG)
    if not os.path.exists(arg.output_dir):
        os.makedirs(arg.output_dir)
    
    output_path = os.path.join(arg.output_dir, 'confusion_matrix.png')
    plt.savefig(output_path, bbox_inches='tight')  # Save image
    print(f'Confusion matrix saved at {output_path}')
