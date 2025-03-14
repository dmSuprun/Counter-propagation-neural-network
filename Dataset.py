import random


class Dataset:
    number_arguments = 2
    number_outputs = 4
    dataset_size = 64

    def __init__(self, range_arguments):
        self.range_arguments = range_arguments
        self.data_x = []
        self.data_y = []
        self.train_x = []
        self.train_y = []
        self.test_1_x = []
        self.test_1_y = []
        self.test_2_x = []
        self.test_2_y = []
        self.__organize_dataset()

    def __fill_up_arguments(self):
        for i in range(self.dataset_size):
            temp = [random.randint(self.range_arguments[0], self.range_arguments[1]),
                    random.randint(self.range_arguments[0], self.range_arguments[1])]
            while temp in self.data_x:
                temp = [random.randint(self.range_arguments[0], self.range_arguments[1]),
                        random.randint(self.range_arguments[0], self.range_arguments[1])]
            self.data_x.append(temp)

    def __fill_up_outputs(self):
        for i in range(self.dataset_size):
            x_1 = self.data_x[i][0]
            x_2 = self.data_x[i][1]
            y_1 = x_1 + x_2
            y_2 = x_1 - x_2
            y_3 = x_1 * x_2
            y_4 = x_1 / x_2
            self.data_y.append([y_1, y_2, y_3, y_4])

    @staticmethod
    def out_data(data1, data2):
        result = [data1[i] + data2[i] for i in range(len(data1))]
        print('_' * 50)
        print(f'Вибірка: ')
        print('_' * 50)
        print(f'Х_1     Х_2     Y_1     Y_2     Y_3     Y_4')
        for i in range(len(result)):
            for j in range(len(result[i])):
                print(result[i][j], end='\t\t')
            print("")

    def out_dataset(self):
        data_1 = [self.train_x[i] + self.train_y[i] for i in range(len(self.train_x))]
        data_2 = [self.test_1_x[i] + self.test_1_y[i] for i in range(len(self.test_1_x))]
        data_3 = [self.test_2_x[i] + self.test_2_y[i] for i in range(len(self.test_2_x))]
        print('_' * 50)
        print(f'Навчальна вибірка: ')
        print('_' * 50)

        print(f'Х_1     Х_2     Y_1     Y_2     Y_3     Y_4')
        for i in range(len(data_1)):
            for j in range(len(data_1[i])):
                print(data_1[i][j], end='\t\t')
            print("")
        print('_' * 50)
        print(f'Контрольна 1 вибірка: ')
        print('_' * 50)
        print(f'Х_1     Х_2     Y_1     Y_2     Y_3     Y_4')
        for i in range(len(data_2)):
            for j in range(len(data_2[i])):
                print(data_2[i][j], end='\t\t')
            print("")
        print('_' * 50)
        print(f'Контрольна 2 вибірка: ')
        print('_' * 50)
        print(f'Х_1     Х_2     Y_1     Y_2     Y_3     Y_4')
        for i in range(len(data_3)):
            for j in range(len(data_3[i])):
                print(data_3[i][j], end='\t\t')
            print("")

    def __organize_dataset(self):
        self.__fill_up_arguments()
        self.__fill_up_outputs()
        self.__slice_data()

    def __slice_data(self):
        self.train_x = self.data_x[:24]
        self.train_y = self.data_y[:24]
        self.test_1_x = self.data_x[24:44]
        self.test_1_y = self.data_y[24:44]
        self.test_2_x = self.data_x[44:]
        self.test_2_y = self.data_y[44:]

    def get_train_data(self):
        return [self.train_x + self.train_y for i in range(len(self.train_x))]

    def get_test_1_x(self):
        return self.test_1_x

    def get_test_1_y(self):
        return self.test_1_y

    def get_test_2_x(self):
        return self.test_2_x

    def get_test_2_y(self):
        return self.test_2_y
