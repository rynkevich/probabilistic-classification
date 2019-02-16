import math
import random
import statistics as stat
import numpy as np
from collections import namedtuple


GaussParameters = namedtuple('GaussParameters', ['mean', 'variance'])
ClassificationErrors = namedtuple('ClassificationErrors', ['false_positive', 'detection_error'])


def generate_vector(mean, standard_deviation, size):
    return [random.gauss(mean, standard_deviation) for _ in range(0, size)]


def probabilistic_classification(probabilities, vectors):
    return [get_probability_density_function(vector, probabilities[i]) for i, vector in enumerate(vectors)]


def get_probability_density_function(vector, probability):
    mean = stat.mean(vector)
    variance = stat.variance(vector)
    return lambda x: calculate_probability_density(x, mean, variance) * probability


def calculate_probability_density(x, mean, variance):
    return math.exp((-(x - mean) ** 2) / 2 * variance) / math.sqrt(2 * math.pi * variance)


def get_error_probabilities(probability_density_functions, interval, epsilon):
    false_positive = 0
    detection_error = 0

    start, end = interval
    for x in np.arange(start, end, epsilon):
        probability_density = [pdf(x) for pdf in probability_density_functions]
        if probability_density[1] < probability_density[0]:
            detection_error += probability_density[1] * epsilon
        else:
            false_positive += probability_density[0] * epsilon

    return ClassificationErrors(false_positive, detection_error)
