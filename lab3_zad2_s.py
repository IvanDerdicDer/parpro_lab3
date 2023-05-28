import numpy as np
from timeit import timeit

def calculate_pi(
        iteration_count=100_000_000
) -> float:
    individual_value_array = np.arange(1, iteration_count + 1)
    individual_value_array = 1 / (1 + np.power((individual_value_array - 0.5) / iteration_count, 2))

    my_pi = 4 * np.sum(individual_value_array) / iteration_count

    print(f"{my_pi:.16f} {my_pi - np.pi:.16f}")

    return my_pi


# print(timeit(lambda: calculate_pi(1000000), number=100) / 100)

calculate_pi(100_000_000)