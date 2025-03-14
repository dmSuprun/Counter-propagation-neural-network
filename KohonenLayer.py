from  KohonenNeuron import KohonenNeuron


class KohonenLayer:
    def __init__(self, number_neurons_in_layer, number_element_in_image, number_x_arg):
        self.neurons = [KohonenNeuron(number_element_in_image, number_x_arg) for i in range(number_neurons_in_layer)]
    def fit_layer(self, image_x, image_y):
        res = [neuron.activate_neuron(image_x, image_y) for neuron in self.neurons]
        temp_max = res[0]
        for i in range(len(res)):
            if res[i] > temp_max:
                temp_max = res[i]
        return [i if temp_max == i else 0 for i in res ]

