import numpy as np
import math

def order_data_by_million(table):
    """
    This function takes in a table and returns a
    sorted table by the number of million in descending order.
    """
    money_index = table.column_names.index("earning_($ million)")
    sorted_table = table.copy()
    for row in table.data:
        pass
    return sorted_table

def get_column(table, header, col_name):
    """ Get a column
    Args:
        table (mypytable): data
        header (list): names in the header
        col_name (str): the column to be pulled out
    Returns:
        col (str): the column to that was requested
    """
    col_index = header.index(col_name)
    col = []
    for row in table:
        value = row[col_index]
        if value != "NA":
            col.append(value)
    return col

def get_frequencies(table, header, col_name):
    """ Get the frequencies in a certain column
    Args:
        table (mypytable): data
        header (list): names in the header
        col_name (str): the column to be pulled out
    Returns:
        values (list): values in each column
        counts (list): number of times each item is in the column
    """
    col = get_column(table, header, col_name)
    for item in col:
        col[col.index(item)] = str(item)
    col.sort() # inplace 
    # parallel lists
    values = []
    counts = []
    for value in col:
        if value in values: # seen it before
            counts[-1] += 1 # okay because sorted
        else: # haven't seen it before
            values.append(value)
            counts.append(1)
    return values, counts

def get_genres(genres: list):
    """ Pull the unique genres from a list
    Args:
        genres (list): full list of genres for each movie
    Returns:
        unique_genres (list): list of unique genres
    """
    unique_genres = []
    for row in genres:
        for item in row:
            if item not in unique_genres:
                unique_genres.append(item)
    return unique_genres

def get_number_counts(lst, x):
    """ Get the frequencies in a certain list for a particular item
    Args:
        lst (list): list of items
        x (int or str): item to count the frequencies of
    Returns:
        count (int): count of items
    """
    count = 0
    for element in lst:
        if (element == x):
            count = count + 1
    return count

def get_percent(lst, x):
    """ Get the percentage of a list
    Args:
        lst (list): list of items
        x (int or str): item to count the frequencies of
    Returns:
        
    """
    return (get_number_counts(lst, x) / len(lst)) * 100

def mean(lst):
    """ Compute the mean of a list
    Args:
        lst (list): list of items
    Returns:
        the mean of the list
    """
    return sum(lst) / len(lst)

def std(input_list):
    """ Compute the standard deviation of a list
    Args:
        input_list (list): list of items
    Returns:
        res (float): standard deviation
    """
    mean = sum(input_list) / len(input_list)
    variance = sum([((x - mean) ** 2) for x in input_list]) / len(input_list)
    res = variance ** 0.5
    return res

def convert_to_float(val_list: list):
    """ Convert percentages to float on a 10 point scale
    Args:
        val_list (list): list to be converted
    Returns:
        converted (list): list that has been converted to float
    """
    converted = []
    for item in range(len(val_list)):
        converted.append(float(val_list[item].replace("%", "")))
    
    return converted

def compute_equal_width_cutoffs(values, num_bins):
    """ Get the cutoff points for data
    Args:
        values (list): list of data
        num_bins (int): number of bins to create
    Returns:
        cutoffs (list): floating point numbers for cutoff values
    """
    values_range = max(values) - min(values)
    bin_width = values_range / num_bins # float
    # range() works well with integer start stop and steps
    # np.arange() is for floating point start stop and steps
    cutoffs = list(np.arange(min(values), max(values), bin_width))
    cutoffs.append(max(values)) # exact max
    # if your application allows, convert cutoffs to ints
    # otherwise optionally round them
    cutoffs = [round(cutoff, 2) for cutoff in cutoffs]
    return cutoffs 

def compute_bin_frequencies(values, cutoffs):
    """ Get the frequencies in a list
    Args:
        values (list): list of data
        cutoffs (list): list of floating point cutoff numbers
    Returns:
        freqs (list): bin frequencies
    """
    freqs = [0 for _ in range(len(cutoffs) - 1)]

    for value in values:
        if value == max(values):
            freqs[-1] += 1 # increment the last bin's freq
        else:
            for i in range(len(cutoffs) - 1):
                if cutoffs[i] <= value < cutoffs[i + 1]:
                    freqs[i] += 1 
    return freqs 

def compute_slope_intercept(x, y):
    """ Compute slope intercept between two points
    Args:
        x (int): x value
        y (int): y value
    Returns:
        m (float): m value 
        b (float): b value
    """
    meanx = np.mean(x)
    meany = np.mean(y)

    num = sum([(x[i] - meanx) * (y[i] - meany) for i in range(len(x))])
    den = sum([(x[i] - meanx) ** 2 for i in range(len(x))])
    m = num / den 
    # y = mx + b => b = y - mx
    b = meany - m * meanx
    return m, b

def compute_correlation(x, y):
    meanx = np.mean(x)
    meany = np.mean(y)
    num = sum([(x[i] - meanx) * (y[i] - meany) for i in range(len(x))])
    den = math.sqrt(sum([(x[i] - meanx) ** 2 for i in range(len(x))]) * sum([(y[i] - meany) ** 2 for i in range(len(x))]))
    return num / den

def compute_covariance(x, y):
    meanx = np.mean(x)
    meany = np.mean(y)
    num = sum([(x[i] - meanx) * (y[i] - meany) for i in range(len(x))])
    den = len(x)
    return num / den

def split_string_count(col: list):
    """ Split strings in a column
    Args:
        column (list): strings that need to be split by ","
    Returns:
        split_column (list of list): the strings split into sublists
    """
    split_column = []
    for row in col:
        split_row = row.split(",")
        split_column.append(split_row)
    #print(split_column)
    return split_column

def divide_ratings(genre_unique, genres, scores):
    """ Get the frequencies in a certain column
    Args:
        
    Returns:
        
    """
    total_scores = []
    for genre in genre_unique:
        scores_per_genre = []
        for row in genres:
            if genre in row:
                scores_per_genre.append(scores[genres.index(row)])
        total_scores.append(scores_per_genre)
    return total_scores