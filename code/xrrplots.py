__author__ = "Alessandro Greco and Christos Karapanagiotis"
__credits__ = ["Vladimir Starostin", "Linus Pithan", "Sascha Liehr", "Alexander Hinderhofer", "Alexander Gerlach",
               "Frank Schreiber", "Stefan Kowarik"]
__version__ = "0.9"

import matplotlib.pyplot as plt
import numpy as np


def plot_reflectivity_gallery(q_vector, measured_intensities, fit_intensity, interval, output='show'):
    number_of_curves = fit_intensity.shape[1]
    plot_range = range(0, number_of_curves, interval)

    number_of_subplots = len(plot_range)

    number_of_columns = 4
    number_of_rows = np.int(np.ceil(number_of_subplots / number_of_columns))

    figure_width = 2.5 * number_of_columns
    figure_height = 2 * number_of_rows

    plt.figure(figsize=[figure_width, figure_height])

    for plot_number in range(number_of_subplots):

        if (plot_number % number_of_columns) == 0:
            is_left = 1
        else:
            is_left = 0

        plt.subplot(number_of_rows, number_of_columns, plot_number + 1)
        curve_index = plot_range[plot_number]
        plot_fit_comparison(q_vector, measured_intensities[:, curve_index], fit_intensity[:, curve_index],
                            ylabel=is_left)

    plt.subplots_adjust(wspace=0.3)

    if output == 'show':
        plt.show()
    elif output == 'save':
        plt.savefig('reflectivity_comparison_gallery.svg')

    plt.close()


def plot_reflectivity(q_vector, intensity_vector, output='none', xlabel=1, ylabel=1):
    plt.semilogy(q_vector * 1e-10, intensity_vector)

    plt.xlabel('q in 1/Å')
    plt.ylabel('Reflectivity')

    if xlabel == 1:
        plt.xlabel('q in 1/Å')
    if ylabel == 1:
        plt.ylabel('Reflectivity')

    if output == 'show':
        plt.show()


def plot_fit_comparison(q_vector, measured_intensity, fit_intensity, output='none', xlabel=1, ylabel=1, legend=1):
    plt.semilogy(q_vector * 1e-10, measured_intensity, 'bo', label='Data')
    plt.semilogy(q_vector * 1e-10, fit_intensity, 'r', label='Fit')

    if legend == 1:
        plt.legend()
    if xlabel == 1:
        plt.xlabel('q in 1/Å')
    if ylabel == 1:
        plt.ylabel('Reflectivity')

    if output == 'show':
        plt.show()


def plot_thickness_vs_time(time_vector, thickness, output='show'):
    plt.figure(figsize=[4, 2])

    plt.plot(time_vector, thickness * 1e+10)

    plt.xlabel('Time')
    plt.ylabel('Thickness in Å')

    plt.tight_layout()

    if output == 'show':
        plt.show()
    elif output == 'save':
        plt.savefig('film_thickness.svg')

    plt.close()


def plot_roughness_vs_time(time_vector, roughness, output='show'):
    plt.figure(figsize=[4, 2])

    plt.plot(time_vector, roughness * 1e+10)

    plt.xlabel('Time')
    plt.ylabel('RMS roughness in Å')

    plt.tight_layout()

    if output == 'show':
        plt.show()
    elif output == 'save':
        plt.savefig('film_roughness.svg')

    plt.close()


def plot_SLD_vs_time(time_vector, scattering_length_density, output='show'):
    plt.figure(figsize=[4, 2])

    plt.plot(time_vector, scattering_length_density * 1e-15)

    plt.xlabel('Time')
    plt.ylabel('SLD in $10^{-5}$ $1/Å^2$')

    plt.tight_layout()

    if output == 'show':
        plt.show()
    elif output == 'save':
        plt.savefig('film_SLD.svg')

    plt.close()
