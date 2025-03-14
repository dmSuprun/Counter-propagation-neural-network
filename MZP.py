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
            outcome = self.kohonen_layer.fit_layer(None, image_y)
        else:
            outcome = self.kohonen_layer.fit_layer(image_x, None)
        return self.grosberg_layer.fit_layer(outcome)

    def train(self, data_train):
        for x in data_train:
            outcome = self.kohonen_layer.train_layer(x)
            self.grosberg_layer.train_layer(x, outcome)

dt = Dataset([2,9])


nw = MZP(2, 4, 2)
nw.train(dt.get_train_data())


print(nw.fit(dt.get_test_1_x()[0],None))
print(dt.get_test_1_y()[0])


