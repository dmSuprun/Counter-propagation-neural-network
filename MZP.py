from random import random
from KohonenLayer import KohonenLayer
from GrosbergLayer import GrosbergLayer
from Dataset import Dataset


class MZP:
    def __init__(self, number_arguments, number_outputs, number_neuron_in_kohonen_layer):
        self.number_arguments = number_arguments
        self.number_outputs = number_outputs
        self.number_element_in_image = number_arguments + number_outputs
        self.kohonen_layer = KohonenLayer(number_neuron_in_kohonen_layer, self.number_element_in_image,
                                          number_arguments)
        self.grosberg_layer = GrosbergLayer(self.number_element_in_image, number_neuron_in_kohonen_layer)

    def fit(self, image_x, image_y):
        if image_x is None:
            outcome = self.kohonen_layer.fit_wta_layer(None, image_y)
        else:
            outcome = self.kohonen_layer.fit_wta_layer(image_x, None)
        return self.grosberg_layer.fit_layer(outcome)

    def train(self, data_train):
        iteration_and_error = {}
        i=1
        for x in data_train:
            outcome = self.kohonen_layer.train_wta_layer(x)
            iteration_and_error[i] = self.grosberg_layer.train_layer(outcome, x)
            i+=1
        print('Мережа завершила своє навчання')
        return iteration_and_error