import argparse
import pickle

import numpy as np
from tqdm import tqdm

Score_1024 = open('pointMLP_SampleMatrix_1024_1_score.pkl', 'rb')
Score_1024 = pickle.load(Score_1024)
label = open('pointMLP_SampleMatrix_1024_1_label.pkl', 'rb')
label = pickle.load(label)

Score_512 = open('pointMLP_SampleMatrix_512_3_score.pkl', 'rb')
Score_512 = pickle.load(Score_512)
# label = open('pointMLP_SampleMatrix_512_3_label.pkl', 'rb')
# label = pickle.load(label)

Score_256 = open('pointMLP_SampleMatrix_256_3_score.pkl', 'rb')
Score_256 = pickle.load(Score_256)
# label = open('pointMLP_SampleMatrix_256_3_label.pkl', 'rb')
# label = pickle.load(label)

print(Score_1024.shape)
print(Score_512.shape)
print(Score_256.shape)
print(label.shape)

import sklearn.metrics as metrics
weights = [0.6, 0.2, 0.2, 0]
test_pred = weights[0] * Score_1024 + weights[1] * Score_512 + weights[2] * Score_256
# test_pred = Score_1024
test_pred = np.argmax(test_pred, axis=1)
print("acc:" + str(float("%.3f" % (100. * metrics.accuracy_score(label, test_pred)))))
print("acc_avg:" + str(float("%.3f" % (100. * metrics.balanced_accuracy_score(label, test_pred)))))


# right_num = total_num = 0
#
# for i in tqdm(range(label.shape[0])):
#     l = label[i]
#     r11 = r1[i]
#     r22 = r2[i]
#     r33 = r3[i]
#     r = 0.6 * r11 + 0.3 * r22 + 0.1 * r33
#     # r = r11 + r22 + r33
#     # r = r33
#     r = np.argmax(r)
#     right_num += int(r == int(l))
#     total_num += 1
# acc = right_num / total_num
# print(acc)