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
        print(outcome)
        return self.grosberg_layer.fit_layer(outcome)

    def train(self, data_train):
        for x in data_train:
            outcome = self.kohonen_layer.train_wta_layer(x)
            print(outcome)
            self.grosberg_layer.train_layer(outcome, x)


dt = Dataset([20, 90])

nw = MZP(2, 4, 6)
dt.normalize()
nw.train(dt.get_train())

for i in range(len(dt.get_test_1_x())):
    print(dt.get_test_1_x()[i], '->', dt.get_test_1_y()[i])
    print(nw.fit(dt.get_test_1_x()[i], None))
print('-' * 100)
for i in range(len(dt.get_test_2_y())):
    print(dt.get_test_2_x()[i], '->', dt.get_test_2_y()[i])
    print(nw.fit(None, dt.get_test_2_y()[i]))
