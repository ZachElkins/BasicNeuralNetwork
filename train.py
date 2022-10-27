import argparse
import json
import time

import utils
from neuralnetwork import NeuralNetwork as NN


def train(nn, number_to_train):
    print("Importing mnist dataset from tensorflow...")
    from tensorflow.keras.datasets import mnist
    (train_data, train_sltn), _ = mnist.load_data()

    training_data = utils.process_data(utils.normalize_data(train_data[:number_to_train], 255))
    training_solutions = utils.one_hot_encoding(train_sltn[:number_to_train], 10)

    print("Starting training...")
    start = time.perf_counter()
    utils.train(nn, training_data, training_solutions)
    end = time.perf_counter()

    print(f"Training took {round(end-start, 2)} seconds\n({round(number_to_train/(end-start), 2)} pictures trained per second)")

    return nn.to_serializable()


def train_and_test(nn, number_to_train, name):
    print("Importing mnist dataset from tensorflow...")
    from tensorflow.keras.datasets import mnist
    (train_data, train_sltn), (test_data, test_sltn) = mnist.load_data()

    number_to_train = number_to_train
    number_to_test = number_to_train

    training_data = utils.process_data(utils.normalize_data(train_data[:number_to_train], 255))
    training_solutions = utils.one_hot_encoding(train_sltn[:number_to_train], 10)
    testing_data = utils.process_data(utils.normalize_data(test_data[:number_to_test], 255))
    testing_solutions = utils.one_hot_encoding(test_sltn[:number_to_test], 10)

    print("Starting training...")
    start = time.perf_counter()
    interval = 100
    x = []
    y = []

    data_len_diff = len(training_data)//len(testing_data)

    for i in range(0, len(training_solutions)-interval, interval):
        # Train
        x.append(i)
        for j in range(interval):
            nn.train(training_data[i+j], training_solutions[i+j])

        errors = 0
        # Predict
        for j in range(interval//data_len_diff):
            pred = list(nn.predict(testing_data[i//data_len_diff+j//data_len_diff]))
            if pred.index(max(pred)) != list(testing_solutions[i//data_len_diff+j//data_len_diff]).index(1):
                errors += 1

        y.append(errors/interval)

    end = time.perf_counter()

    print(f"Training took {round(end-start, 2)} seconds\n({round(number_to_train/(end-start), 2)} pictures trained per second)")

    utils.plot_many(data=[{'x': x, 'y': y, 'name': name}], min_line=False, max_line=False)

    return nn.to_serializable()


def main(args):
    nn = NN(
        input_nodes=784,
        hidden_nodes=args.hn,
        output_nodes=10,
        activation_function=args.a.lower(),
        learning_rate=args.lr
    )

    if args.g:
        model = train_and_test(nn, args.t, args.n)
    else:
        model = train(nn, args.t)

    with open(f'{args.n}.json', 'w') as f:
        json.dump(model, f)

    print(f"Model saved as {args.n}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Neural Network on the mnist dataset."
    )

    parser.add_argument('-hn', action="store", type=int, required=False,
                        default=700,
                        help="The number of hidden nodes to use. (Default: 700)")
    parser.add_argument('-a', action="store", type=str, required=False,
                        default="Sigmoid",
                        help="The activation function to use. Sigmoid, Tanh, ReLU. (Default: Sigmoid)")
    parser.add_argument('-lr', action="store", type=float, required=False,
                        default=0.05,
                        help="The learning rate. Must be within the exclusive range (0, 1). (Default: 0.05)")
    parser.add_argument('-t', action="store", type=int, required=False,
                        default=30000,
                        help="The number of images used to train the model in the range (0, 60000]. (Default: 30000)")
    parser.add_argument('-g', action="store_true", required=False,
                        help="Flag to graph the training over time. (This will add significant time to training)")
    parser.add_argument('-n', action="store", type=str, required=False,
                        default="model",
                        help="Filename, not including extension. (Default: model)")

    args = parser.parse_args()

    if args.t <= 0 or args.t > 60000:
        raise ValueError("You must train on at least 1 item, or at most 60000 items.")

    main(args)
