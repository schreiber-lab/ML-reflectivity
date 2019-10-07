__author__ = "Alessandro Greco and Christos Karapanagiotis"
__credits__ = ["Vladimir Starostin", "Linus Pithan", "Sascha Liehr", "Alexander Hinderhofer", "Alexander Gerlach",
               "Frank Schreiber", "Stefan Kowarik"]
__version__ = "0.9"

import csv

import numpy as np
from tqdm import tqdm

import config_loader
import reflectivity as refl

config = config_loader.ConfigLoader('organic.config')

test_data_file_name = config.get_test_data_file_name()
test_data = np.loadtxt(test_data_file_name)  # in 1/m
q_values = test_data[:, 0] * 1e10

noise_factor = 0

training_data_name, training_labels_name, validation_data_name, validation_labels_name = config.get_training_file_names()

number_of_training_samples, number_of_validation_samples = config.get_number_of_training_samples()

min_thickness, max_thickness = config.get_thickness()

min_roughness, max_roughness = config.get_roughness()

min_scattering_length_density, max_scattering_length_density = config.get_scattering_length_density()


def main():
    np.random.seed(config.get_random_seed())

    [training_data, training_labels] = make_training_data(number_of_training_samples)
    [validation_data, validation_labels] = make_training_data(number_of_validation_samples)

    save_data_as_file(training_data, training_data_name)
    save_data_as_file(validation_data, validation_data_name)

    save_labels_as_file(training_labels, training_labels_name)
    save_labels_as_file(validation_labels, validation_labels_name)


def make_training_data(n_samples):
    """Randomly generate n_samples sets of parameters and curves for training or validation."""
    training_data_input = make_training_input(n_samples)
    [thicknesses, roughnesses, SLDs] = training_data_input

    training_reflectivity = make_reflectivity_curves(q_values, thicknesses, roughnesses, SLDs, n_samples)

    training_data_output = np.zeros([len(q_values), n_samples + 1])
    training_data_output[:, 0] = q_values
    training_data_output[:, 1:] = training_reflectivity

    return training_data_output, training_data_input


def make_reflectivity_curves(q_values, thicknesses, roughnesses, SLDs, number_of_curves):
    """Generate and return reflectivity curves for given q_values and film parameters."""
    reflectivity_curves = np.zeros([len(q_values), number_of_curves])

    for curve in tqdm(range(number_of_curves)):
        reflectivity = refl.multilayer_reflectivity(q_values, thicknesses[curve, :],
                                                    roughnesses[curve, :], SLDs[curve, :])

        reflectivity_noisy = apply_shot_noise(reflectivity)

        reflectivity_curves[:, curve] = reflectivity_noisy

    return reflectivity_curves


def apply_shot_noise(reflectivity_curve):
    """Apply shot noise to a reflectivity curve depending on noise_factor and return noisy curve."""
    noisy_reflectivity = np.clip(np.random.normal(reflectivity_curve, noise_factor * np.sqrt(reflectivity_curve)), 1e-8,
                                 None)

    return noisy_reflectivity


def make_training_input(number_of_sets):
    """Randomly generate and return film parameters for number_of_sets reflectivity curves."""
    randomized_thicknesses = randomize_inputs(min_thickness, max_thickness, number_of_sets)

    randomized_roughnesses = np.zeros_like(randomized_thicknesses)

    for sample in range(number_of_sets):
        for layer in range(randomized_thicknesses.shape[1]):
            max_roughness_by_thickness = 0.5 * randomized_thicknesses[sample, layer]

            random_roughness_by_thickness = randomize_inputs([min_roughness[layer]], [max_roughness_by_thickness], 1)

            if random_roughness_by_thickness > max_roughness[layer]:
                randomized_roughnesses[sample, layer] = max_roughness[layer]
            elif random_roughness_by_thickness < min_roughness[layer]:
                randomized_roughnesses[sample, layer] = min_roughness[layer]
            else:
                randomized_roughnesses[sample, layer] = random_roughness_by_thickness

    randomized_SLDs = randomize_inputs(min_scattering_length_density, max_scattering_length_density, number_of_sets)

    return randomized_thicknesses, randomized_roughnesses, randomized_SLDs


def randomize_inputs(min_value, max_value, number_of_samples):
    """Return an array of size number_of_samples containing random values between min_value and max_value."""
    min_value = np.asarray(min_value)
    max_value = np.asarray(max_value)

    if np.all(np.isreal(min_value)) and np.all(np.isreal(max_value)):

        number_of_layers = len(min_value)

        randomized_inputs = np.zeros([number_of_samples, number_of_layers])

        for layer in range(number_of_layers):
            randomized_inputs[:, layer] = np.random.uniform(min_value[layer], max_value[layer], number_of_samples)

        return randomized_inputs

    else:
        real_min_value = min_value.real
        real_max_value = max_value.real

        imag_min_value = min_value.imag
        imag_max_value = max_value.imag

        real_randomized_inputs = randomize_inputs(real_min_value, real_max_value, number_of_samples)
        imag_randomized_inputs = randomize_inputs(imag_min_value, imag_max_value, number_of_samples)

        complex_randomized_inputs = real_randomized_inputs + 1j * imag_randomized_inputs

        return complex_randomized_inputs


def save_data_as_file(dataset, file_name):
    """Save dataset as .txt file with name file_name."""
    number_of_samples = dataset.shape[1] - 1
    header = []

    for i in range(number_of_samples):
        sample = i + 1
        header += ['Reflectivity {}'.format(sample)]

    header = ['q [1/m]'] + header

    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f, dialect=csv.excel_tab)
        writer.writerow(header)
        writer.writerows(dataset)


def save_labels_as_file(labels, file_name):
    """Save labels as .txt file with name file_name."""
    thicknesses = labels[0]
    roghnesses = labels[1]
    SLDs = labels[2]

    labels = np.concatenate((thicknesses, roghnesses, SLDs), axis=1)

    number_of_layers = thicknesses.shape[1]

    thickness_header = []
    roughness_header = []
    SLD_header = []

    for i in range(number_of_layers):
        layer = i + 1
        thickness_header += ['Thickness {} [m]'.format(layer)]
        roughness_header += ['Roughness {} [m^2]'.format(layer)]
        SLD_header += ['Scattering length density {} [1/m^2]'.format(layer)]

    header = thickness_header + roughness_header + SLD_header

    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f, dialect=csv.excel_tab)
        writer.writerow(header)
        writer.writerows(labels)


if __name__ == '__main__':
    main()
