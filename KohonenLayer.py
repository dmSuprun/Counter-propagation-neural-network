from KohonenNeuron import KohonenNeuron
import math


class KohonenLayer:
    def __init__(self, number_neurons_in_layer, number_element_in_image, number_x_arg):
        self.neurons = [KohonenNeuron(number_element_in_image, number_x_arg) for i in range(number_neurons_in_layer)]

    def fit_wta_layer(self, image_x, image_y):
        res = [neuron.activate_neuron(image_x, image_y) for neuron in self.neurons]
        t = 0
        temp_max = res[t]
        for i in range(len(res)):
            if res[i] > temp_max:
                temp_max = res[i]
                t = i
        res = len(res) * [0]
        res[t] = 1
        return res

    def fit_soft_max_layer(self, image_x, image_y):
        return [neuron.activate_neuron(image_x, image_y) for neuron in self.neurons]

    # def calculate_distance(self, neuron1, neuron2):
    #     # Можна використовувати евклідову відстань між нейронами
    #     return math.sqrt(sum([(neuron1.get_weight()[i] - neuron2.get_weight()[i]) ** 2 for i in range(len(neuron1.get_weight()))]))
    #
    # def train_wta_layer_with_neighborhood(self, image, neighborhood_radius):
    #     # Знайдемо нейрон-переможець
    #     res = [neuron.activate_all_image(image) for neuron in self.neurons]
    #     temp_ind = 0
    #     temp_max = res[temp_ind]
    #     for i in range(len(res)):
    #         if res[i] > temp_max:
    #             temp_max = res[i]
    #             temp_ind = i
    #     target_neuron = self.neurons[temp_ind]
    #
    #     # Оновлюємо ваги переможця та його сусідів
    #     for i, neuron in enumerate(self.neurons):
    #         # Обчислюємо відстань між переможцем і поточним нейроном
    #         distance = self.calculate_distance(target_neuron, neuron)
    #
    #         if distance <= neighborhood_radius:
    #             # Чим ближче нейрон, тим більше він навчається
    #             influence = math.exp(-distance / (2 * (neighborhood_radius ** 2)))
    #             target_weights = neuron.get_weight()
    #             for j in range(len(target_weights)):
    #                 target_weights[j] += neuron.learning_rate * influence * (image[j] - target_weights[j])
    #             neuron.set_weight(target_weights)
    #
    #             # Зменшуємо навчальну швидкість
    #             if neuron.learning_rate > 0.1:
    #                 neuron.learning_rate -= 0.1
    #
    #     # Повертаємо результати
    #     res = len(res) * [0]
    #     res[temp_ind] = 1
    #     return res
    #

    def train_wta_layer(self, image):
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
        if target_neuron.learning_rate > 0.1:
            target_neuron.learning_rate -= 0.1
        res = len(res) * [0]
        res[temp_ind] = 1
        return res

    def train_soft_max_layer(self, image):
        res = [neuron.activate_all_image(image) for neuron in self.neurons]
        t = sum(res)
        res = [r / t for r in res]

        for neuron in self.neurons:
            target_weights = neuron.get_weight()
            for i in range(len(target_weights)):
                target_weights[i] += neuron.learning_rate * (image[i] - target_weights[i])
            neuron.set_weight(target_weights)
            if neuron.learning_rate > 0.01:
                neuron.learning_rate -= 0.01

        return res
