def get_index(generator_index, hour_index, n_hours) -> int:
    """# Find composite index into 1D list for (generator, hour)"""
    return generator_index * n_hours + hour_index

def get_generator_and_day(index, hours) -> tuple:
    """Inverse of get_index - given a composite index in a 1D list 
    Return the generator and hour"""
    generator_index, hour_index = divmod(index, hours)
    return generator_index, hour_index