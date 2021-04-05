import argparse
import os

import numpy as np


def main(args):
    path = args.path
    all_models = [os.path.join(path, dI) for dI in os.listdir(path) if os.path.isdir(os.path.join(path, dI))]
    all_results = []
    for model in all_models:
        with open(model + "/result.txt", "r") as f:
            content = f.readlines()[:-1]
            content = [x.strip() for x in content]
        result_subtasks = [float(x[x.find(":") + 1 :].strip()) for x in content]
        all_results += result_subtasks
    all_results = np.array(all_results) * 100
    mean = all_results.mean()
    std_of_means = all_results.std() / np.sqrt(len(all_results))
    print("Mean: " + str(mean))
    print("Std of means: " + str(std_of_means))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute control statistics")

    parser.add_argument("--path", required=True, type=str, help="path to ilqr result")
    args = parser.parse_args()

    main(args)
