# Find composite index into 1D list for (generator, hour)
def get_index(generator_index, hour_index, n_hours) -> int:
    return generator_index * n_hours + hour_index

# Inverse of get_index - given a composite index in a 1D list, return the
# generator and hour
def get_generator_and_day(index, hours) -> tuple:
    generator_index, hour_index = divmod(index, hours)
    return generator_index, hour_index