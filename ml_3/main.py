import pandas as pd
import numpy as np


# Посчитать все объекты с размерами больше определенных
# Посчитать все объекты со средней скоростью больше определенной
# Найти все объекты, которые проходили через определенную зону кадра
# Найти все объекты, которые не двигались дольше N кадров подряд
# Определить кадры, когда в кадре было более N объектов одновременно
# Определить кадры, когда не было объектов
# Рассчитать для всех объектов среднее время нахождения в кадре ( время жизни объекта)
# Построить heatmap - визуализацию, как много объектов было в конкретном участке кадра ( чем больше объектов было в этом участке - тем краснее)
# Определить объекты, которые за N кадров сдвинулись не более чем на M пикселей


# 1. Посчитать все объекты с размерами больше определенных
def get_objects(data, x, y):
    def get_size(object):
        x_r = object['x_right']
        y_b = object['y_bottom']
        x_l = object['x_left']
        y_t = object['y_top']
        id = object['ID']
        return np.abs(x_r - x_l), np.abs(y_b - y_t), id

    size = get_size(data)
    arr = np.asarray(size).T
    id = []
    for x_obj, y_obj, id_obj in arr:
        if x_obj > x and y_obj > y:
            id.append(id_obj)
    return np.unique(id)


# 5. Определить кадры, когда в кадре было более N объектов одновременно
def get_frames_greater_N(data, N):
    uniq_frame = np.unique(data['frame'])  # уникальные кадры
    amount_objects = data['frame'].value_counts(sort=False)  # просто количество объектов в кадре
    arr = np.asarray(amount_objects)
    f = []  # кадры, где количество объектов в фрейме >N

    for i in uniq_frame - 1:
        if arr[i] > N:
            f.append(i)

    return f


# 6. Определить кадры, когда не было объектов
def get_frames_without_objects(data):
    uniq_frame = np.unique(data['frame'])

    no_obj = []

    all_frames = np.array(range(1, uniq_frame.size + 1))

    for i in all_frames:
        if i not in uniq_frame:
            no_obj.append(i)

    return no_obj


# Найти все объекты, которые проходили через определенную зону кадра

def get_objects_zone(data, x_right, x_left, y_up, y_low):
    def get_size(object):
        x_r = object['x_right']
        y_b = object['y_bottom']
        x_l = object['x_left']
        y_t = object['y_top']
        id = object['ID']
        return x_r, x_l, y_b, y_t, id

    size = get_size(data)
    arr = np.asarray(size).T
    id = []
    for x_r_obj, y_b_obj, x_l_obj, y_t_obj, id_obj in arr:
        if x_right > x_r_obj > x_left and x_right > x_l_obj > x_left and y_up > y_b_obj > y_low and y_up > y_t_obj > y_low:
            id.append(id_obj)
    return np.unique(id)


if __name__ == '__main__':
    data = pd.read_csv("MyProject/trajectories.csv", sep=';')

    # Посчитать все объекты с размерами больше определенных

    count_obj = get_objects(data, 100, 100)

    print(f'Объекты с размерами больше определенных: {len(count_obj)}\n')

    # Определить кадры, когда в кадре было более N объектов одновременно
    N = 15
    frames = get_frames_greater_N(data, N)

    print(f'Кадры, когда в кадре было более {N} объектов: {frames}\n Size: {len(frames)}\n')

    # Определить кадры, когда не было объектов
    no_object_frames = get_frames_without_objects(data)
    print(f'Кадры, когда не было объектов: {no_object_frames}\n')

    # Найти все объекты, которые проходили через определенную зону кадра
    id = get_objects_zone(data, 700, 100, 600, 100)
    print(f'Найти все объекты, которые проходили через определенную зону кадра: Size: {len(id)}\n')
