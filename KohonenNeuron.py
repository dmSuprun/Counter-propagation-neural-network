class KohonenNeuron:
    learning_rate = 0.7

    def __init__(self, number_element_in_image, number_x_arg):
        self.__weights = [1 / (number_element_in_image ** 0.5) for i in range(number_element_in_image)]
        self.__number_x_arg = number_x_arg

    def activate_neuron(self, image_x, image_y):
        activation = 0
        weights_for_y = self.weights[self.number_x_arg:]
        weights_for_x = self.weights[:self.number_x_arg]
        if image_x is None:
            for i in range(len(weights_for_y)):
                activation += image_y[i] * weights_for_y[i]

        elif image_y is None:
            for i in range(len(weights_for_x)):
                activation += image_x[i] * weights_for_x[i]
        else:
            for i in range(len(weights_for_x)):
                activation += image_x[i] * weights_for_x[i]
            for i in range(len(weights_for_y)):
                activation += image_y[i] * weights_for_y[i]
        return activation

    def activate_all_image(self, image):
        activation = 0
        for i in range(len(image)):
            activation += image[i] * self.weights[i]
        return activation

    def get_weight(self):
        return self.weights

    def set_weight(self, weight):
        self.weights = weight
