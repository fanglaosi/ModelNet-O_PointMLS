import pickle
import argparse
import numpy as np
from tqdm import tqdm

def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--test_path', default='test_score', help='path to save testing score')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    Score_1024 = open('{}/PointMLS_basic_score.pkl'.format(args.test_path), 'rb')
    Score_1024 = pickle.load(Score_1024)
    label = open('{}/PointMLS_basic_label.pkl'.format(args.test_path), 'rb')
    label = pickle.load(label)

    Score_512 = open('{}/PointMLS_512_score.pkl'.format(args.test_path), 'rb')
    Score_512 = pickle.load(Score_512)

    Score_256 = open('{}/PointMLS_256_score.pkl'.format(args.test_path), 'rb')
    Score_256 = pickle.load(Score_256)

    Score_128 = open('{}/PointMLS_128_score.pkl'.format(args.test_path), 'rb')
    Score_128 = pickle.load(Score_128)

    import sklearn.metrics as metrics
    weights = [0.6, 0.2, 0.2, 0.2]
    test_pred = weights[0] * Score_1024 + weights[1] * Score_512 + weights[2] * Score_256 + weights[3] * Score_128
    test_pred = np.argmax(test_pred, axis=1)
    print("==================================== ensemble results ====================================")
    print("acc:" + str(float("%.3f" % (100. * metrics.accuracy_score(label, test_pred)))))
    print("acc_avg:" + str(float("%.3f" % (100. * metrics.balanced_accuracy_score(label, test_pred)))))

