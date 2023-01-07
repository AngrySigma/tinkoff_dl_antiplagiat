"""
Compare file pairs for plagiarism and save scores to file
Example usage: python compare.py ../data/input.txt ../data/score.txt --model ../models/model.pkl
"""

from argparse import ArgumentParser
import pickle


parser = ArgumentParser()
parser.add_argument('input')
parser.add_argument('scores')
parser.add_argument('--model', default=False)


def load_model(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def compare_files(initial_file, suspect_file):
    with open(initial_file, 'r', encoding='utf-8') as f:
        initial_text = f.read()
    with open(suspect_file, 'r', encoding='utf-8') as f:
        suspect_text = f.read()
    return 1


def save_scores(scores, file):
    with open(file, 'w') as f:
        for score in scores:
            f.write(f'{score}\n')


if __name__ == '__main__':
    # parse arguments
    args = parser.parse_args()
    # load model
    model = load_model(args.model)
    # set scores and compare files
    scores = []
    with open(args.input, 'r') as file:
        lines = file.readlines()
        for line in lines:
            initial_file, suspect_file = line.split()
            scores.append(compare_files(initial_file, suspect_file))
    # save scores
    save_scores(scores, args.scores)
