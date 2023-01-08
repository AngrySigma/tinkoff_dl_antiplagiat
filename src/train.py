"""
Train script.Saves model to file.
Exapmle usage: python train.py ../data/files ../data/plagiat1 ../data/plagiat2 --model ../models/model.pkl
"""

from argparse import ArgumentParser
import os
import pickle
import torch
import math

parser = ArgumentParser()
parser.add_argument("files")
parser.add_argument("plagiat1")
parser.add_argument("plagiat2")
parser.add_argument("--model", default=False)


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
    with open(args.model, "wb") as f:
        pickle.dump(model, f)


def train_model():
    pass


def get_dataset():
    pass


class Model:
    def __init__(self, network, optimizer, loss_fn, dataset, lr=0.001):
        self.network = network
        self.network.to('cuda')
        self.optimizer = optimizer(self.network.parameters(), lr=lr)
        self.loss_fn = loss_fn(reduction="mean")

    def fit(self, dataloader, epochs):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1} of {epochs}")
            average_loss = torch.tensor([0]).float().to('cuda')
            for count, batch in enumerate(dataloader):
                loss = torch.tensor([0]).float().to('cuda')
                for sample in batch:
                    features = sample[0].unsqueeze(0)
                    target = sample[1].unsqueeze(0)
                    out = self.network(features)
                    loss += self.loss_fn(out, target.float())
                loss /= dataloader.batch_size
                average_loss += loss
                if count % int(len(dataloader) / 10) == 0:
                    print(f'{int(count / len(dataloader) * 100)}%')
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            average_loss /= len(dataloader)
            print(f"Epoch: {epoch}, Average loss: {average_loss.item()}")

        return self


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dir_path, plagiat1_path, plagiat2_path):
        self.files = get_files_in_all_dirs(
            dir_path, plagiat1_path, plagiat2_path
        )
        self.dir_path = os.path.join(*os.path.split(dir_path))
        self.plagiat1_path = os.path.join(*os.path.split(plagiat1_path))
        self.plagiat2_path = os.path.join(*os.path.split(plagiat2_path))

    def __len__(self):
        len_files = len(self.files)
        return int(len_files * (len_files - 1) / 2 + len_files * 2)

    def get_filenames_by_index(self, index):
        """
        Return filenames of dataset item by index.
        Assuming N files have N(N-1)/2 pairs.

        :param index: index of dataset item
        :return: dataset item
        """
        # plagiarized file pairs
        len_files = len(self.files)
        valid_pairs = int(len_files * (len_files - 1) / 2)
        if index >= valid_pairs:
            index -= valid_pairs
            if index < len_files:
                file2 = os.path.join(self.plagiat1_path, self.files[index])
            else:
                index -= len_files
                file2 = os.path.join(self.plagiat2_path, self.files[index])
            file1 = os.path.join(self.dir_path, self.files[index])
            mark = 1
        else:
            # normal file pairs
            num1 = math.floor((1 + math.sqrt(1 + 8 * index)) / 2)
            num2 = int(index - num1 * (num1 - 1) / 2)
            file1 = os.path.join(self.dir_path, self.files[num1])
            file2 = os.path.join(self.dir_path, self.files[num2])
            mark = 0
        return file1, file2, mark

    @staticmethod
    def length_comparison(file1, file2):
        """
        Stupid method to start learning and get some features.
        Compare lengths of files

        :param file1: path to file1
        :param file2: path to file2
        :return: length of file1 divided by length of file2
        """
        with open(file1, "r", encoding="utf-8") as f:
            text1 = f.read()
        with open(file2, "r", encoding="utf-8") as f:
            text2 = f.read()
        return len(text1) / max(1, len(text2))

    def __getitem__(self, index):
        file1, file2, mark = self.get_filenames_by_index(index)
        features = []
        features.append(self.length_comparison(file1, file2))
        features.append(mark)
        features = torch.FloatTensor(features).to('cuda')
        return features


if __name__ == "__main__":
    args = parser.parse_args()
    dataset = Dataset(args.files, args.plagiat1, args.plagiat2)
    # to get 1 plagiarized pair in batch on average use batch of 32
    batch_size = 128
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    network = torch.nn.Sequential(
        torch.nn.Linear(1, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1),
        torch.nn.Sigmoid(),
    )
    model = Model(network=network, dataset=dataset, optimizer=torch.optim.Adam,
                  loss_fn=torch.nn.BCELoss, lr=0.001)
    # model trains on cuda by default
    model.fit(dataloader, epochs=10)
    save_model(model)
