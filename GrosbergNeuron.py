import random


class GrosbergNeuron:
    learning_rate = 0.7

    def __init__(self, number_neurons_in_previous_layer):
        self.weights = [random.uniform(0, 1) for _ in range(number_neurons_in_previous_layer)]

    def activate_neuron(self, input_image):
        activation = 0
        for i in range(len(input_image)):
            activation += self.weights[i] * input_image[i]
        return activation

    def train_neuron(self, input_image, output_image):
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * (output_image[i] - self.weights[i]) * input_image[i]
