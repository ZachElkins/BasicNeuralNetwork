import numpy as np
from matplotlib import pyplot as plt


def one_hot_encoding(data_set, size):
    '''
        Converts a one-dimensional integer array in to a 2D
        array of the one-hot representation of each value.
        ex: one_hot_encoding([0, 1, 2], 3) => [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    '''
    if isinstance(data_set, int):
        one_hot = np.zeros(size)
        one_hot[data_set] = 1
        return one_hot

    new_data_set = []
    for _, data in enumerate(data_set):
        one_hot = np.zeros(size)
        one_hot[data] = 1
        new_data_set.append(one_hot)

    return new_data_set


def one_hot_decoding(data_set):
    if not isinstance(data_set[0], list):
        return data_set.index(max(data_set))

    new_data_set = []
    for i, data in enumerate(data_set):
        new_data_set.append(data.index(max(data)))

    return new_data_set


def normalize_data(data_set, normalizing_factor):
    '''Function to normalize the data between 0 and 1'''
    data_set = np.array(data_set)
    if normalizing_factor == 0:
        return data_set.astype(np.float128)
    return data_set.astype(np.float128) / normalizing_factor


def process_data(data_set):
    '''Function for processing all the number images in to 1-dimensional lists'''
    new_data_set = []
    for data in data_set:
        processed = data.flatten()
        new_data_set.append(normalize_data(processed, processed.sum()))

    return new_data_set


def train(
    nn,
    training_data,
    training_sltn
):
    for data, sltn in zip(training_data, training_sltn):
        nn.train(data, sltn)


def test_err(
    nn,
    testing_data,
    testing_sltn
):
    errors = 0
    for data, sltn in zip(testing_data, testing_sltn):
        pred = nn.predict(data).tolist()
        if pred.index(max(pred)) != list(sltn).index(1):
            errors += 1

    return errors / len(testing_data)


def plot_many(data, points=True, regresssion=True, min_line=True, max_line=True):
    '''
    Plot the training data as a scatter plot,
    with optinoal additionoal information
    '''

    if not any([points, regresssion, min_line, max_line]):
        points = True

    colors = [
        {'light': 'red', 'dark': 'darkred'},
        {'light': 'blue', 'dark': 'midnightblue'},
        {'light': 'limegreen', 'dark': 'darkolivegreen'},
        {'light': 'mediumorchid', 'dark': 'indigo'},
        {'light': 'orange', 'dark': 'peru'},
        {'light': 'grey', 'dark': 'black'},
    ]
    legend = []

    _, ax = plt.subplots(nrows=1, ncols=1)

    max_y = -1
    for i, data_point in enumerate(data):
        if max(data_point['y']) > max_y:
            max_y = max(data_point['y'])

        if points:
            ax.scatter(data_point['x'], data_point['y'], s=2, alpha=0.65, color=colors[i]['light'])
            legend.append(f'{data_point["name"]} points')

        if regresssion:
            linear_reg_fn1 = np.poly1d(np.polyfit(data_point['x'], data_point['y'], 1))
            ax.plot(data_point['x'], linear_reg_fn1(data_point['x']), '-', color=colors[i]['dark'])
            legend.append(f'{data_point["name"]} regression')

        if min_line:
            ax.plot(data_point['x'], np.zeros(len(data_point['y']))+min(data_point['y']), '--', color=colors[i]['dark'], alpha=0.4, label='_nolegend_')
        if max_line:
            ax.plot(data_point['x'], np.zeros(len(data_point["y"]))+max(data_point['y']), '--', color=colors[i]['dark'], alpha=0.4, label='_nolegend_')

    names = ""
    for data_point in data[1:]:
        names += f" vs. {data_point['name']}"
    ax.set_title(f'{data[0]["name"]}{names} Error Rate Over Time')
    ax.set_xlabel('Number of Images Used in Training')
    ax.set_ylabel('Error Rate')
    ax.set_ylim([0, max_y+0.1])
    ax.legend(legend)

    plt.show()
