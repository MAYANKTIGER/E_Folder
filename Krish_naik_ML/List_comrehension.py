lst1 = []
def sqr_root(lst):
    for i in lst:
        lst1.append(i*i)
    return lst1

print(sqr_root([1,2,3,4,4]))


lst = [1,2,3,4,5]
lst2 = [i*i for i in lst if i%2==0]
print(lst2)
