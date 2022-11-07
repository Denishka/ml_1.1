import pandas as pd
import numpy as np
import re


def get_matrix(dataset, size_matrix):
    array_states = np.array(dataset.split(';'))

    matrix = np.full((size_matrix, size_matrix), 0.0)
    for i in range(len(array_states) - 1):
        matrix[int(array_states[i]), int(array_states[i + 1])] += 1

    for i in range(size_matrix):
        if matrix[i].sum() == 0:
            continue
        matrix[i] = matrix[i] / matrix[i].sum()
    return matrix


def get_result(prob_matrix, dataset_for_user, threshold=0.1):
    anomalies = []

    dataset_for_user = np.array(dataset_for_user.split(';'))

    for i in range(len(dataset_for_user) - 1):
        if (0 > int(dataset_for_user[i])) and (int(dataset_for_user[i]) >= len(prob_matrix)):
            anomalies.append(0)

        if (prob_matrix[int(dataset_for_user[i]), int(dataset_for_user[i + 1])]) > threshold:
            anomalies.append(1)
        else:
            anomalies.append(0)

    return anomalies


def change_data_fake(data_fake):
    indexes = []
    for i in range(0, len(data_fake)):
        indexes.append(i)

    data_fake['INDEX'] = indexes
    data_fake.set_index('INDEX', inplace=True)
    data_fake = data_fake['DATA']
    return data_fake


if __name__ == '__main__':
    data = pd.read_csv('datasets/data.txt', sep=':')
    data_fake = pd.read_csv('datasets/data_fake.txt', sep=':')
    data_true = pd.read_csv('datasets/data_true.txt', sep=':')

    new_array = data['DATA']
    # new_array2= data['USER'][0]

    data_true = data_true['DATA']

    dataset_united = ';'.join(new_array)
    entire_dataset_array = np.array(dataset_united.split(';')).astype(int)
    unique_dataset = set(entire_dataset_array)

    size_matrix = max(set(unique_dataset)).astype(int) + 1

    a = []
    for i in range(new_array.size):
        a.append(get_matrix(new_array[i], size_matrix))

    array_true = []
    for i in range(data_true.size):
        array_true.append(get_result(a[i], data_true[i]))

    data_fake = data_fake.sort_values(by='USER', key=lambda val: val.str.replace('user', '').astype(int))

    data_fake = change_data_fake(data_fake)
    array_fake = []

    for i in range(data_fake.size):
        array_fake.append(get_result(a[i], data_fake[i]))

    for i in range(data_fake.size):
        print(f'user{i + 1} {array_fake[i]}')

    fake_sum = 0
    sum_all = 0
    for user in array_fake:
        fake_sum += sum(user)
        sum_all += len(user)
    print(f'Data fake: {1 - fake_sum / sum_all}')

    fake_sum = 0
    sum_all = 0
    for user in array_true:
        fake_sum += sum(user)
        sum_all += len(user)
    print(f'Data true: {fake_sum / sum_all}')
