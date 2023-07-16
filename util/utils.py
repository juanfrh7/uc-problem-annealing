import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_index(generator_index, hour_index, n_hours) -> int:
    """
    Find composite index into 1D list for (generator, hour)

    Args:
        generator_index (int): The index of the generator.
        hour_index (int): The index of the hour.
        n_hours (int): The total number of hours.

    Returns:
        int: The composite index representing the position of the (generator, hour) pair in a 1D list.
    """
    return generator_index * n_hours + hour_index


def get_generator_and_day(index, hours) -> tuple:
    """
    Inverse of get_index - given a composite index in a 1D list, return the generator and hour.

    Args:
        index (int): The composite index representing the position in the 1D list.
        hours (int): The total number of hours.

    Returns:
        tuple: A tuple containing the generator index and hour index corresponding to the given composite index.
               The tuple has the form (generator_index, hour_index).
    """
    generator_index, hour_index = divmod(index, hours)
    return generator_index, hour_index

def calculate_relative_error(classical_results, quantum_results, total_bits):

    # Flatten the lists of positions
    classical_positions = [pos for sublist in classical_results for pos in sublist]
    quantum_positions = [pos for sublist in quantum_results for pos in sublist]

    error_count = len(set(quantum_positions) - set(classical_positions))
    relative_error = error_count / total_bits

    return relative_error