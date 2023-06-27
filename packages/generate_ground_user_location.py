import numpy as np


def generate_GU_xyz(
    num_gu=5,
    x_min=0,
    x_max=100,
    y_min=0,
    y_max=100
):
    # generate random x-axis
    gu_x = np.random.uniform(low=x_min, high=x_max, size=(num_gu,))

    # generate random y-axis
    gu_y = np.random.uniform(low=y_min, high=y_max, size=(num_gu,))

    # generate z-axis
    gu_z = np.zeros((num_gu,))

    # combine xyz into array
    gu_xyz = np.vstack([gu_x, gu_y, gu_z]).T

    return gu_xyz
