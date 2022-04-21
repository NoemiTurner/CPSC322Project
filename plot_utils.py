from turtle import width
import matplotlib.pyplot as plt
from sympy import rotations
import utils
import random
import numpy as np
import math

def generate_frequency_diagram_bar(xs: list, ys: list, title: str, x_label: str, y_label: str):
    """ Generates a frequency diagram
    Args:
        xs(list): x data
        ys (list): y data
    """
    figure = plt.figure(figsize=(11, 4))
    plt.title(title)
    ax = figure.add_axes([0,0,1,1])
    ax.bar(xs, ys, width=.8)
    plt.xticks(rotation=45)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def generate_pie_chart(xs: list, ys: list):
    """ Generates a pie chart
    Args:
        xs (list): x data
        ys (list): y data
    """
    plt.pie(ys, labels=xs, autopct="%1.1f%%")
    plt.axis
    fig1, ax1 = plt.subplots()
    ax1.pie(ys, labels=xs, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

def generate_histograms(data1: list, x_label: str, y_label: str, title: str, length=11, width=4):
    """ Generates a histogram
    Args:
        data1 (list): data to generate chart
        x_label (str): x data label
        y_label (str): y data label
        title (str): title of chart
    """
    plt.figure(figsize=(length, width))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=50)
    plt.title('Histogram of ' + title)
    plt.hist(data1, bins=10) # default is 10
    plt.grid(True)
    plt.show()

def generate_scatterplots(xs: list, ys: list, x_label: str, y_label: str):
    """ Generates a scatterplot
    Args:
        xs (list): x data
        ys (list): y data
        x_label (str): x data label
        y_label (str): y data label
    """
    plt.figure(figsize=(11, 4))
    plt.scatter(xs, ys)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    m, b = utils.compute_slope_intercept(xs, ys)
    print("m:", m, "b:", b)
    covariance = utils.compute_covariance(xs, ys)
    correlation = utils.compute_correlation(xs, ys)
    plt.plot([min(xs), max(xs)], [m * min(xs) + b, m * max(xs) + b], c="r", lw=5, label="cov: " + str(covariance) + " corr: " + str(correlation))
    plt.legend()
    plt.show()

def generate_discretization(values: list, bins: int, x_label: str, y_label: str):
    """ Generates discretization
    Args:
        values (list): data
        bins (int) : number of bins to use
        x_label (str): x data label
        y_label (str): y data label
    """
    plt.figure(figsize=(11, 4))
    cutoffs = utils.compute_equal_width_cutoffs(values, bins)
    print(cutoffs)
    freqs = utils.compute_bin_frequencies(values, cutoffs)
    print(freqs)
    plt.xticks(cutoffs)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.bar(cutoffs[:-1], freqs, width=(cutoffs[1] - cutoffs[0]), 
        edgecolor="black", align="edge")

def generate_box_plot(distributions, labels, x_label, y_label, title):
    """ Generates box plot
    Args:
        distributions (list of list): values
        labels (list): x data
        x_label (str): x data label
        y_label (str): y data label
        title (str): title of the chart
    """
    # distributions and labels are parallel
    # distributions: list of 1D lists of values
    plt.figure(figsize=(11, 4))
    plt.boxplot(distributions)
    # boxes correspond to the 1st and 3rd quartiles
    # line in the middle of the box corresponds to the 2nd quartile (AKA median)
    # whiskers corresponds to +/- 1.5 * IQR
    # IQR: interquartile range (3rd quartile - 1st quartile)
    # circles outside the whiskers correspond to outliers
    
    # customize x ticks
    plt.xticks(list(range(1, len(distributions) + 1)), labels)
    plt.xticks(rotation=45)
    # annotations
    # we want to add "mu=100" to the center of our figure
    # xycoords="data": default, specify the location of the label in the same
    # xycoords = "axes fraction": specify the location of the label in absolute
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
