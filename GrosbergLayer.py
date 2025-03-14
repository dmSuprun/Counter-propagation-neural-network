from GrosbergNeuron import GrosbergNeuron


class GrosbergLayer:
    def __init__(self, number_neuron_in_layer, number_neuron_in_previous_layer):
        self.neurons = [GrosbergNeuron(number_neuron_in_previous_layer) for _ in range(number_neuron_in_layer)]

    def train_layer(self, input_image, output_image):
        for i in range(len(self.neurons)):
            target = self.neurons[i]
            target.train_neuron(input_image, output_image)
            if target.learning_rate > 0.01:
                target.learning_rate -= 0.01

    def fit_layer(self, input_image):
        return [self.neurons[i].activate_neuron(input_image) for i in range(len(self.neurons))]
