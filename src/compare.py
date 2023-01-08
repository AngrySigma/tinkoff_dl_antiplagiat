"""
Compare file pairs for plagiarism and save scores to file
Example usage: python compare.py ../data/input.txt ../data/score.txt --model ../models/model.pkl
"""

from argparse import ArgumentParser
from train import Model, compare_files_imports, compare_files_classes, Dataset
import torch
import ast

parser = ArgumentParser()
parser.add_argument("input")
parser.add_argument("scores")
parser.add_argument("--model", default=False)


def compare_files(initial_file, suspect_file):
    with open(initial_file, "r", encoding="utf-8") as f:
        initial_text = f.read()
    with open(suspect_file, "r", encoding="utf-8") as f:
        suspect_text = f.read()
    tree1 = ast.parse(initial_text)
    tree2 = ast.parse(suspect_text)
    imports_distance = compare_files_imports(
        initial_text, suspect_text, tree1, tree2
    )
    classes_distance = compare_files_classes(
        initial_text, suspect_text, tree1, tree2
    )
    lengths = Dataset.length_comparison(initial_text, suspect_text)
    data = torch.FloatTensor([lengths, imports_distance, classes_distance]).to(
        "cuda"
    )
    return model.network(data).item()


def save_scores(scores, file):
    with open(file, "w") as f:
        for score in scores:
            f.write(f"{score}\n")


if __name__ == "__main__":
    args = parser.parse_args()
    model = torch.load(args.model)
    # set scores and compare files
    scores = []
    with open(args.input, "r") as file:
        lines = file.readlines()
        for line in lines:
            initial_file, suspect_file = line.split()
            scores.append(compare_files(initial_file, suspect_file))
    save_scores(scores, args.scores)
