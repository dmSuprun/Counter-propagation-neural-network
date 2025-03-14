from KohonenNeuron import KohonenNeuron


class KohonenLayer:
    def __init__(self, number_neurons_in_layer, number_element_in_image, number_x_arg):
        self.neurons = [KohonenNeuron(number_element_in_image, number_x_arg) for i in range(number_neurons_in_layer)]

    def fit_layer(self, image_x, image_y):
        res = [neuron.activate_neuron(image_x, image_y) for neuron in self.neurons]
        temp_max = res[0]
        for i in range(len(res)):
            if res[i] > temp_max:
                temp_max = res[i]
        return [1 if temp_max == i else 0 for i in res]

    def learn_layer(self, image):
        res = [neuron.activate_all_image(image) for neuron in self.neurons]
        temp_ind = 0
        temp_max = res[temp_ind]
        for i in range(len(res)):
            if res[i] > temp_max:
                temp_max = res[i]
                temp_ind = i
        target_neuron = self.neurons[temp_ind]
        target_weights = target_neuron.get_weight()
        for i in range(len(target_weights)):
            target_weights[i] += target_neuron.learning_rate * (image[i] - target_weights[i])
        target_neuron.set_weight(target_weights)
        if target_neuron.learning_rate > 0.01:
            target_neuron.learning_rate -= 0.01
        return [1 if temp_max == i else 0 for i in res]
