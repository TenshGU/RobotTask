import numpy as np

Dict = {}
coordinate = Dict.get(1) if Dict.__contains__(1) else []
coordinate.append([1, 2])
Dict.setdefault(1, coordinate)
print(Dict)

PRICE = [[1, 2]]


def init(Price: []):
    Price.append([1, 2])


p = []
init(p)
print(p)

a = 0b00000110
i = 3
print(a & (1 << i))

if 3 and 0:
    print(1111111)

# class test:
#     def __init__(self, dest: np.ndarray):
#         self.d = dest
#
#
# t = test(np.array([1, 2]))
# print(t.__dict__)
# print(np.array([1, 2]))
#
# w = test(np.array([1, 2]))
# k = test(np.array([1, 2]))
# s = [w]
# if w in s:
#     print(1)
# elif k in s:
#     print(2)
#
#
# for i in range(1, 10):
#     print(i)
#
# for i in range(2, 10):
#     print(i)
#
# print(float('inf') > 1000)