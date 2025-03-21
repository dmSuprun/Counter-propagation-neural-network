import random


class Dataset:
    number_arguments = 2
    number_outputs = 4
    dataset_size = 64
    normal = []

    def __init__(self, range_arguments):
        self.range_arguments = range_arguments
        self.data = []
        self.train = []
        self.test_1 = []
        self.test_2 = []
        self.__organize_dataset()

    def __fill_up_arguments(self):
        for i in range(self.dataset_size):
            temp = [random.randint(self.range_arguments[0], self.range_arguments[1]),
                    random.randint(self.range_arguments[0], self.range_arguments[1])]
            while temp in self.data:
                temp = [random.randint(self.range_arguments[0], self.range_arguments[1]),
                        random.randint(self.range_arguments[0], self.range_arguments[1])]
            self.data.append(temp)

    def __fill_up_outputs(self):
        for i in range(self.dataset_size):
            x_1 = self.data[i][0]
            x_2 = self.data[i][1]
            y_1 = x_1 + x_2
            self.data[i].append(y_1)
            y_2 = x_1 - x_2
            self.data[i].append(y_2)
            y_3 = x_1 * x_2
            self.data[i].append(y_3)
            y_4 = x_1 / x_2
            self.data[i].append(y_4)

    def __organize_dataset(self):
        self.__fill_up_arguments()
        self.__fill_up_outputs()
        self.__slice_data()


    def normalize(self):
        for i in range(self.dataset_size):
            temp = sum([el ** 2 for el in self.data[i]]) ** 0.5
            self.normal.append(temp)
            for j in range(len(self.data[i])):
                self.data[i][j] = self.data[i][j] / temp
        self.__slice_data()

    def denormalize(self):
        for i in range(self.dataset_size):
            for j in range(len(self.data[i])):
                self.data[i][j] *= self.normal[i]
        self.__slice_data()

    def denormalize_test_1_pred(self, pred):
        res_n = self.normal[24:44]
        for i in range(len(pred)):
            for j in range(len(pred[i])):
                pred[i][j] *= res_n[i]
        return pred

    def denormalize_test_2_pred(self, pred):
        res_n = self.normal[44:]
        for i in range(len(pred)):
            for j in range(len(pred[i])):
                pred[i][j] *= res_n[i]
        return pred

    def __slice_data(self):
        self.train = self.data[:24]
        self.test_1 = self.data[24:44]
        self.test_2 = self.data[44:]


    def output_non_normal_data(self,data):
        print(f'Х_1\t\t\tХ_2\t\t\tY_1\t\t\tY_2\t\t\tY_3\t\t\tY_4')
        for i in range(len(data)):
            for j in range(len(data[i])):
                print(f"{data[i][j]:.2f}",end='\t\t')
            print("")

    def output_x(self, dataX):
        print(f'\t\tX_1\t\t\t\t\t\tX_2')
        for i in range(len(dataX) - 1):
            print(f'{dataX[i][0]}\t\t{dataX[i][1]}')

    def output_y(self, dataY):
        print(f'\t\tY_1\t\t\t\t\t\tY_2\t\t\t\t\t\tY_3\t\t\t\t\t\tY_4')
        for i in range(len(dataY) - 1):
            print(f'{dataY[i][0]}\t\t{dataY[i][1]}\t\t{dataY[i][2]}\t\t{dataY[i][3]}')

    def output_data(self, dat):
        print(f'\t\tX_1\t\t\t\t\t\tX_2\t\t\t\t\t\tY_1\t\t\t\t\t\tY_2\t\t\t\t\t\tY_3\t\t\t\t\t\tY_4')
        for i in range(len(dat) - 1):
            print(f'{dat[i][0]}\t\t{dat[i][1]}\t\t{dat[i][2]}\t\t{dat[i][3]}\t\t{dat[i][4]}\t\t{dat[i][5]}')

    def get_train(self):
        return self.train

    def get_test_1_x(self):
        x = []
        for i in range(len(self.test_1)):
            x.append(self.test_1[i][:self.number_arguments])
        return x

    def get_test_1_y(self):
        y = []
        for i in range(len(self.test_1)):
            y.append(self.test_1[i][self.number_arguments:])
        return y

    def get_test_2_x(self):
        x = []
        for i in range(len(self.test_2)):
            x.append(self.test_2[i][:self.number_arguments])
        return x

    def get_test_2_y(self):
        y = []
        for i in range(len(self.test_2)):
            y.append(self.test_2[i][self.number_arguments:])
        return y

    def get_test_1(self):
        return self.test_1

    def get_test_2(self):
        return self.test_2
