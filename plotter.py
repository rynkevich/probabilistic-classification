import matplotlib.pyplot as plt
import numpy as np


FUNCTION_LABELS = (r'$p(\frac{X_m}{C_1})P(C_1)$', r'$p(\frac{X_m}{C_2})P(C_2)$')
AREA_LABELS = (r'Detection error of $C_1$', r'False positive of $C_1$')


def display_results(probability_density_functions, error_probabilities, interval, epsilon):
    start, end = interval
    range_to_display = np.arange(start, end, epsilon)

    y = tuple(map(lambda pdf: np.array([pdf(x) for x in range_to_display]), probability_density_functions))

    _, ax = plt.subplots()
    ax.plot(range_to_display, y[0], label=FUNCTION_LABELS[0])
    ax.plot(range_to_display, y[1], label=FUNCTION_LABELS[1])

    ax.fill_between(range_to_display, 0, y[0], where=y[1] >= y[0], label=AREA_LABELS[0])
    ax.fill_between(range_to_display, 0, y[1], where=y[0] >= y[1], label=AREA_LABELS[1])

    info = '\n'.join([
        r'P("False positive") = %.2f' % error_probabilities.false_positive,
        r'P("Detection error") = %.2f' % error_probabilities.detection_error,
        r'P("Error") = %.2f' % sum(error_probabilities)
    ])
    ax.text(ax.get_xlim()[0] * 0.95, ax.get_ylim()[1] * 0.95, info,
            fontsize='12', verticalalignment='top',
            bbox={'boxstyle': 'round', 'alpha': 0.5})

    plt.legend(loc='upper right', fontsize='12')
    plt.title('Probabilistic Classification')

    plt.show()
