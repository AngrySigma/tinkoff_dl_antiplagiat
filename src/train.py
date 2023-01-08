"""
Train script.Saves model to file.
Example usage:
python train.py <files dir> <plagiat dir> <plagiat dir> --model <model file.pt>
Dirs used in project: ../data/files, ../data/plagiat1, ../data/plagiat2
"""

from argparse import ArgumentParser
import os
import torch
import math
import ast

parser = ArgumentParser()
parser.add_argument("files")
parser.add_argument("plagiat1")
parser.add_argument("plagiat2")
parser.add_argument("--model", default=False)


def change_row(matrix, i):
    matrix[i] = [math.inf for _ in range(len(matrix[i]))]


def get_column(matrix, j):
    column = []
    for i in range(len(matrix)):
        column.append(matrix[i][j])
    return column


def change_column(matrix, j):
    for i in range(len(matrix)):
        matrix[i][j] = math.inf


def remove_row(matrix, i):
    matrix.pop(i)


def remove_column(matrix, j):
    for i in range(len(matrix)):
        matrix[i].pop(j)


def remove_from_nodes(nodes, i):
    nodes.pop(i)


def get_node_source(source, node):
    if isinstance(node, ast.ClassDef):
        return node.name
    else:
        return ast.get_source_segment(source, node)


def compare_similar_nodes(source1, source2, nodes1, nodes2):
    result_dict = {}
    similarity_matrix = [
        [0 for _ in range(len(nodes2))] for _ in range(len(nodes1))
    ]
    for i, node1 in enumerate(nodes1):
        for j, node2 in enumerate(nodes2):
            similarity_matrix[i][j] = edit_distance(
                get_node_source(source1, node1),
                get_node_source(source2, node2),
            )
    i = 0
    while len(similarity_matrix) > 0 and len(similarity_matrix[0]) > 0:
        row = similarity_matrix[i]
        row_min_index = min(range(len(row)), key=row.__getitem__)
        column = get_column(similarity_matrix, row_min_index)
        column_min_index = min(range(len(column)), key=column.__getitem__)
        if column_min_index == i:
            result_dict[nodes1[i]] = (
                nodes2[row_min_index],
                similarity_matrix[i][row_min_index],
            )
            remove_row(similarity_matrix, i)
            remove_column(similarity_matrix, row_min_index)
            remove_from_nodes(nodes1, i)
            remove_from_nodes(nodes2, row_min_index)
            if i >= len(similarity_matrix):
                i = 0
        else:
            i = (i + 1) % len(similarity_matrix)
    return result_dict


def get_node_types(tree):
    nobody_types = {}
    body_types = {}
    for i in range(len(tree.body)):
        node = tree.body[i]
        if hasattr(node, "body"):
            if body_types.get(type(node)) is None:
                body_types[type(node)] = [node]
            else:
                body_types[type(node)].append(node)
        else:
            if nobody_types.get(type(node)) is None:
                nobody_types[type(node)] = [node]
            else:
                nobody_types[type(node)].append(node)
    return nobody_types, body_types


def get_types_distance(comparison_result, types=None):
    average_distance = 0
    n = 0
    for key, values in comparison_result.items():
        if types is not None and key in types:
            for node in values:
                average_distance += values[node][1]
                n += 1
        elif types is None:
            for node in values:
                average_distance += values[node][1]
                n += 1
        else:
            continue
    return average_distance / n if n != 0 else 0


def compare_files_imports(source1, source2, initial_tree, suspect_tree):
    initial_nobody_types, initial_body_types = get_node_types(initial_tree)
    suspect_nobody_types, suspect_body_types = get_node_types(suspect_tree)
    comparison_result = {}
    for key, value in initial_nobody_types.items():
        if (
            suspect_nobody_types.get(key) is not None
            and len(value) > 0
            and len(suspect_nobody_types[key]) > 0
        ):
            comparison_result[key] = compare_similar_nodes(
                source1, source2, value, suspect_nobody_types[key]
            )
    return get_types_distance(comparison_result, [ast.Import, ast.ImportFrom])


def get_class_defs(tree, class_defs):
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_defs.append(node)
        if hasattr(node, "body"):
            get_class_defs(node, class_defs)


def get_function_defs(tree, function_defs):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            function_defs.append(node)
        if hasattr(node, "body"):
            get_function_defs(node, function_defs)


def compare_files_classes(source1, source2, initial_tree, suspect_tree):
    initial_tree_classes = []
    suspect_tree_classes = []
    get_class_defs(initial_tree, initial_tree_classes)
    get_class_defs(suspect_tree, suspect_tree_classes)
    comparison_result = {
        0: compare_similar_nodes(
            source1, source2, initial_tree_classes, suspect_tree_classes
        )
    }
    return get_types_distance(comparison_result)


def compare_files_functions(source1, source2, initial_tree, suspect_tree):
    initial_tree_functions = []
    suspect_tree_functions = []
    get_function_defs(initial_tree, initial_tree_functions)
    get_function_defs(suspect_tree, suspect_tree_functions)
    comparison_result = {
        0: compare_similar_nodes(
            source1, source2, initial_tree_functions, suspect_tree_functions
        )
    }
    return get_types_distance(comparison_result)


def edit_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for index2, char2 in enumerate(s2):
        new_distances = [index2 + 1]
        for index1, char1 in enumerate(s1):
            if char1 == char2:
                new_distances.append(distances[index1])
            else:
                new_distances.append(
                    1
                    + min(
                        (
                            distances[index1],
                            distances[index1 + 1],
                            new_distances[-1],
                        )
                    )
                )
        distances = new_distances
    return distances[-1]


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


class Model:
    """
    Class for model, contains network, optimizer and methods for training
    """

    def __init__(self, network, optimizer, loss_fn, lr=0.001):
        self.network = network
        self.network.to("cuda")
        self.optimizer = optimizer(self.network.parameters(), lr=lr)
        self.loss_fn = loss_fn(reduction="mean")

    def fit(self, dataloader, epochs):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1} of {epochs}")
            average_loss = torch.tensor([0]).float().to("cuda")
            for count, batch in enumerate(dataloader):
                # if zero tensor is in batch, skip it
                if torch.all(
                    batch == torch.FloatTensor([0, 0, 0, 0, 0]).to("cuda")
                ):
                    continue
                loss = torch.tensor([0]).float().to("cuda")
                for sample in batch:
                    features = sample[:-1].unsqueeze(0)
                    target = sample[-1].unsqueeze(0)
                    out = self.network(features)[0]
                    loss += self.loss_fn(out, target.float())
                loss /= dataloader.batch_size
                average_loss += loss
                if count % int(len(dataloader) / 100) == 0:
                    print(f"{int(count / len(dataloader) * 100)}%")
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            average_loss /= len(dataloader)
            print(f"Epoch: {epoch}, Average loss: {average_loss.item()}")
        return self


class Dataset(torch.utils.data.Dataset):
    """
    Dataset class for pytorch. Takes paths to dirs with files.
    Sample is a pair of files.
    Having different files, any pair is not plagiarized.
    Plagiarised pairs are created as pairs of files from different dirs.
    """

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
    def length_comparison(text1, text2):
        return len(text1) / max(len(text2), 1)

    @staticmethod
    def get_text(file1, file2):
        with open(file1, "r", encoding="utf-8") as f:
            text1 = f.read()
        with open(file2, "r", encoding="utf-8") as f:
            text2 = f.read()
        return text1, text2

    def __getitem__(self, index):
        """
        Take filenames, process and create features.
        Features are:
        1. Length comparison
        2. Levenshtein distance in imports
        3. Levenshtein distance in Class names
        4. Levenshtein distance in function names
        Mark is 1 if files are plagiarized, 0 otherwise

        :param index:
        :return:
        """
        file1, file2, mark = self.get_filenames_by_index(index)
        text1, text2 = self.get_text(file1, file2)
        # if file cannot be parsed, return zero tensor to skip it
        try:
            tree1 = ast.parse(text1)
            tree2 = ast.parse(text2)
        except Exception:
            return torch.FloatTensor([0, 0, 0, 0, 0]).to("cuda")
        features = [
            self.length_comparison(text1, text2),
            compare_files_imports(text1, text2, tree1, tree2),
            compare_files_classes(text1, text2, tree1, tree2),
            compare_files_functions(text1, text2, tree1, tree2),
            mark,
        ]
        features = torch.FloatTensor(features).to("cuda")
        return features


if __name__ == "__main__":
    args = parser.parse_args()
    dataset = Dataset(args.files, args.plagiat1, args.plagiat2)
    # 1 plagiarized pair on average in batch of 32
    batch_size = 128
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    network = torch.nn.Sequential(
        torch.nn.Linear(4, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 1),
        torch.nn.Sigmoid(),
    )
    model = Model(
        network=network,
        optimizer=torch.optim.Adam,
        loss_fn=torch.nn.BCELoss,
        lr=0.001,
    )
    # model trains on cuda by default
    model.fit(dataloader, epochs=3)
    torch.save(model, args.model)
