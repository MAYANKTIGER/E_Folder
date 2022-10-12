
from numpy import iterable


def even_or_odd(num):
    if num%2==0:
        return "the {} is even".format(num) 

    else:
        return "the {}  is odd".format(num)




lst = [1,2,3,4,5,5,6]

print(list(map(even_or_odd, lst)))

#map function mao the define function and the iterable

