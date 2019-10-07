import csv
import time

import keras
import numpy as np

import config_loader
import generate_training_data
import training
import xrrplots

config = config_loader.ConfigLoader('organic.config')


def main():
    # load the model
    model_name = config.get_model_name()
    # model_name = 'model_run1.hdf5'

    custom_object_dict = dict([('abs_err_%d' % (i + 1), training.y_absolute_error(i + 1)) for i in range(4)])
    custom_object_dict['abs_err'] = training.y_absolute_error(0)
    model = keras.models.load_model(model_name, custom_objects=custom_object_dict)

    # import your test data
    path_test_data = config.get_test_data_file_name()
    # path_test_data = 'interpolated_log_reflectivity_SLS_RT.txt'
    test_data = np.loadtxt(path_test_data, delimiter='\t')

    q_vector = test_data[:, 0] * 1e10
    test_data = test_data[:, 1:]

    test_input = np.transpose(test_data)
    test_scores = np.log(test_input)

    time_before_prediction = time.perf_counter()
    predicted_labels = model.predict(test_scores)
    time_after_prediction = time.perf_counter()

    number_of_curves = predicted_labels.shape[0]

    labels = restore_labels(predicted_labels)

    number_of_layers = int(labels.shape[1] / 3)

    thicknesses = labels[:, 0:number_of_layers]
    roughnesses = labels[:, number_of_layers:2 * number_of_layers]
    SLDs = labels[:, 2 * number_of_layers:3 * number_of_layers]

    predicted_reflectivity = np.zeros([len(q_vector), number_of_curves + 1])
    predicted_reflectivity[:, 0] = q_vector

    predicted_reflectivity = generate_training_data.make_reflectivity_curves(q_vector, thicknesses, roughnesses, SLDs,
                                                                             number_of_curves)

    predicted_reflectivity = np.concatenate((np.reshape(q_vector, (q_vector.shape[0], 1)), predicted_reflectivity),
                                            axis=1)

    save_predicted_labels_as_file(labels)
    save_predicted_reflectivity_as_file(predicted_reflectivity)

    total_prediction_time = time_after_prediction - time_before_prediction
    time_per_curve = total_prediction_time / number_of_curves
    print(f'Pure prediction time for {number_of_curves} curves: {total_prediction_time * 1000:.2f} ms.\n')
    print(f'Prediction time per curve: {time_per_curve * 1000:.2f} ms.\n')

    xrrplots.plot_reflectivity_gallery(q_vector, test_data, predicted_reflectivity[:, 1:], 10, output='show')

    xrrplots.plot_thickness_vs_time(range(number_of_curves), thicknesses[:, 0], output='show')

    xrrplots.plot_roughness_vs_time(range(number_of_curves), roughnesses[:, 0], output='show')

    xrrplots.plot_SLD_vs_time(range(number_of_curves), SLDs[:, 0], output='show')


def insert_oxide_thickness(path_train_data, labels):
    train_data = np.loadtxt(path_train_data, skiprows=1)

    labels = np.insert(labels, 1, train_data[:, 1])

    return labels


def restore_labels(labels):
    min_thickness, max_thickness = config.get_thickness()
    min_roughness, max_roughness = config.get_roughness()
    min_scattering_length_density, max_scattering_length_density = config.get_scattering_length_density()

    min_labels = min_thickness + min_roughness + min_scattering_length_density
    max_labels = max_thickness + max_roughness + max_scattering_length_density

    number_of_labels = len(max_labels)

    for label in range(number_of_labels):
        if min_labels[label] == max_labels[label]:
            labels = np.insert(labels, label, max_labels[label], axis=1)
        else:
            labels[:, label] = labels[:, label] * (max_labels[label] - min_labels[label]) + min_labels[label]

    restored_labels = labels

    return restored_labels


def save_predicted_labels_as_file(predicted_labels):
    number_of_layers = int(predicted_labels.shape[1] / 3)

    thickness_header = []
    roughness_header = []
    SLD_header = []

    for i in range(number_of_layers):
        layer = i + 1
        thickness_header += ['Thickness {} [m]'.format(layer)]
        roughness_header += ['Roughness {} [m^2]'.format(layer)]
        SLD_header += ['Scattering length density {} [1/m^2]'.format(layer)]

    header = thickness_header + roughness_header + SLD_header

    with open('prediction_labels.txt', 'w', newline='') as f:
        writer = csv.writer(f, dialect=csv.excel_tab)
        writer.writerow(header)
        writer.writerows(predicted_labels)


def save_predicted_reflectivity_as_file(predicted_reflectivity):
    number_of_samples = predicted_reflectivity.shape[1] - 1
    header = []

    for i in range(number_of_samples):
        sample = i + 1
        header += ['Reflectivity {}'.format(sample)]

    header = ['q [1/m]'] + header

    with open('predictions.txt', 'w', newline='') as f:
        writer = csv.writer(f, dialect=csv.excel_tab)
        writer.writerow(header)
        writer.writerows(predicted_reflectivity)


if __name__ == '__main__':
    main()
