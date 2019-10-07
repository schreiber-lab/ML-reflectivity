__author__ = "Alessandro Greco and Christos Karapanagiotis"
__credits__ = ["Vladimir Starostin", "Linus Pithan", "Sascha Liehr", "Alexander Hinderhofer", "Alexander Gerlach",
               "Frank Schreiber", "Stefan Kowarik"]
__version__ = "0.9"

import configobj


class ConfigLoader:
    """Class for loading settings from a .config file."""

    def __init__(self, config_file_name):
        self.config_object = configobj.ConfigObj(config_file_name)

    def get_test_data_file_name(self):
        test_data_file_name = self.config_object['training']['test_data_file_name']

        return test_data_file_name

    def get_training_file_names(self):
        training_data_file_name = self.config_object['training']['training_data_file_name']
        training_labels_file_name = self.config_object['training']['training_labels_file_name']

        validation_data_file_name = self.config_object['training']['validation_data_file_name']
        validation_labels_file_name = self.config_object['training']['validation_labels_file_name']

        return training_data_file_name, training_labels_file_name, validation_data_file_name, validation_labels_file_name

    def get_random_seed(self):
        random_seed = int(self.config_object['training']['random_seed'])

        return random_seed

    def get_number_of_training_samples(self):
        number_of_training_samples = int(self.config_object['training']['number_of_training_samples'])
        number_of_validation_samples = int(self.config_object['training']['number_of_validation_samples'])

        return number_of_training_samples, number_of_validation_samples

    def get_model_name(self):
        model_name = self.config_object['training']['model_name']

        return model_name

    def get_number_of_epochs(self):
        number_of_epochs = int(self.config_object['training']['number_of_epochs'])

        return number_of_epochs

    def get_thickness(self):
        min_thickness = self._convert_to_list_of_floats(self.config_object['film-properties']['min_thickness'])
        max_thickness = self._convert_to_list_of_floats(self.config_object['film-properties']['max_thickness'])

        return min_thickness, max_thickness

    def get_roughness(self):
        min_roughness = self._convert_to_list_of_floats(self.config_object['film-properties']['min_roughness'])
        max_roughness = self._convert_to_list_of_floats(self.config_object['film-properties']['max_roughness'])

        return min_roughness, max_roughness

    def get_scattering_length_density(self):
        min_scattering_length_density = self._convert_to_list_of_floats(
            self.config_object['film-properties']['min_scattering_length_density'])
        max_scattering_length_density = self._convert_to_list_of_floats(
            self.config_object['film-properties']['max_scattering_length_density'])

        return min_scattering_length_density, max_scattering_length_density

    @staticmethod
    def _convert_to_list_of_floats(list_of_strings):
        if type(list_of_strings) == str:
            return [float(list_of_strings)]
        else:
            list_of_floats = [float(item) for item in list_of_strings]

        return list_of_floats
