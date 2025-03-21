from Dataset import Dataset
from MZP import MZP
import matplotlib.pyplot as plt

dataset = Dataset([2, 9])
train = dataset.get_train()
test_1_x = dataset.get_test_1_x()
test_1_y = dataset.get_test_1_y()
test_2_x = dataset.get_test_2_x()
test_2_y = dataset.get_test_2_y()
print('_' * 20)
print(f'Навчальна вибірка')
print('_' * 20)
dataset.output_non_normal_data(train)
print('_' * 20)
print(f'Контрольна-1 вибірка')
print('_' * 20)
dataset.output_non_normal_data(dataset.get_test_1())
print('_' * 20)
print(f'Контрольна-2 вибірка')
print('_' * 20)

dataset.output_non_normal_data(dataset.get_test_2())
dataset.normalize()
train = dataset.get_train()
test_1_x = dataset.get_test_1_x()
test_1_y = dataset.get_test_1_y()
test_1 = dataset.get_test_1()
test_2 = dataset.get_test_2()
test_2_x = dataset.get_test_2_x()
test_2_y = dataset.get_test_2_y()
print('_' * 20)
mzp_network = MZP(2, 4, 2)
data = mzp_network.train(train)
print('_' * 20)
print(f'Тестування мережі\nПодання на вхід лише X')
outcome1 = []
for x in test_1_x:
    outcome1.append(mzp_network.fit(x, None))

print('_' * 20)
print(f'Тестування мережі\nПодання на вхід лише Y')
outcome2 = []
for y in test_2_y:
    outcome2.append(mzp_network.fit(None, y))

# er=0
#
# for i in range(len(test_1_x)):
#     for j in range(len(test_1[i])):
#         er+= (test_1[i][j] - outcome[i][j])**2
# print(er)
# outcome = dataset.denormalize_test_1_pred(outcome)
# dataset.output_non_normal_data(outcome)


# iterations = list(data.keys())
# errors = list(data.values())
#
# plt.plot(iterations, errors, marker='o', linestyle='-', color='b')
#
# plt.xlabel("Ітерація")
# plt.ylabel("Помилка")
# plt.title("Залежність помилки від ітерації")
# plt.grid(True)
#
# plt.show()
