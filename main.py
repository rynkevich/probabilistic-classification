import math
from sys import argv
from probabilistic_classification import GaussParameters, \
    probabilistic_classification, get_error_probabilities, generate_vector
from plotter import display_results


VECTOR_SIZE = 10000
EPSILON = 0.001


def main():
    parameters = (
        GaussParameters(mean=float(argv[1]), variance=float(argv[2])),
        GaussParameters(mean=float(argv[3]), variance=float(argv[4]))
    )
    probabilities = (float(argv[5]), 1 - float(argv[5]))

    vectors = [generate_vector(params.mean, math.sqrt(params.variance), VECTOR_SIZE) for params in parameters]
    interval = (
        min(map(lambda vector: min(vector), vectors)),
        max(map(lambda vector: max(vector), vectors))
    )

    probability_density_functions = probabilistic_classification(probabilities, vectors)
    error_probabilities = get_error_probabilities(probability_density_functions, interval, EPSILON)

    display_results(probability_density_functions, error_probabilities, interval, EPSILON)


if __name__ == '__main__':
    main()
