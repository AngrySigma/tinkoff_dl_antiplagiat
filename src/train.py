"""
Train script.Saves model to file.
Exapmle usage: python train.py ../data/files ../data/plagiat1 ../data/plagiat2 --model ../models/model.pkl
"""

from argparse import ArgumentParser
import os
import pickle

parser = ArgumentParser()
parser.add_argument('files')
parser.add_argument('plagiat1')
parser.add_argument('plagiat2')
parser.add_argument('--model', default=False)


def get_files_in_all_dirs(path1, path2, path3):
    """
    Get files all dirs have in common
    :param path1: path to dir1
    :param path2: path to dir2
    :param path3: path to dir3
    :return: list of files
    """
    files1 = os.listdir(path1)
    files2 = os.listdir(path2)
    files3 = os.listdir(path3)
    return [file for file in files1 if file in files2 and file in files3]


def save_model(model):
    with open(args.model, 'wb') as f:
        pickle.dump(model, f)


def train_model():
    pass


def get_dataset():
    pass


class Model:
    pass


if __name__ == '__main__':
    args = parser.parse_args()
    train_files = get_files_in_all_dirs(args.files, args.plagiat1, args.plagiat2)
    train_path = os.path.split(args.files)
    plagiat1_path = os.path.split(args.plagiat1)
    plagiat2_path = os.path.split(args.plagiat2)
    model = Model()
    save_model(model)

    for file in train_files:
        print(os.path.join(*train_path, file))
