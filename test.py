import argparse
import codecs
import json
import time

import utils
from neuralnetwork import NeuralNetwork as NN


def test(nn, number_to_test):
    print("Importing mnist dataset from tensorflow...")
    from tensorflow.keras.datasets import mnist
    _, (test_data, test_sltn) = mnist.load_data()

    testing_data = utils.process_data(utils.normalize_data(test_data[:number_to_test], 255))
    testing_solutions = utils.one_hot_encoding(test_sltn[:number_to_test], 10)

    print("Starting tests...")
    start = time.perf_counter()
    err = utils.test_err(nn, testing_data, testing_solutions)
    end = time.perf_counter()

    print(f"Error rate: {round(err*100, 2)}%\n(Testing took {round(end-start, 2)} seconds)\n({round(number_to_test/(end-start), 2)} tests per second)")


def main(args):
    nn_dict = json.loads(codecs.open(f'{args.f}.json', 'r', encoding='utf-8').read())
    nn = NN.from_dict(nn_dict)
    test(nn, args.t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a Neural Network created with train.py"
    )

    parser.add_argument('-t', action="store", type=int, required=False,
                        default=1000,
                        help="The number of images used to test the model in the range (0, 10000]. (Default: 1000)")
    parser.add_argument('-f', action="store", type=str, required=False,
                        default="model",
                        help="Filename of model to test, not including extension. (Default: model)")

    args = parser.parse_args()

    if args.t <= 0 or args.t > 10000:
        raise ValueError("You must train on at least 1 item, or at most 10000 items.")

    main(args)
