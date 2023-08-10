#지금 테스트 하는 중입니다.
#지금 테스트 하는 중입니다.



my_dict = {'a': 1, 'b': 2, 'c': 3}

print(my_dict.items()) #
print(my_dict.keys())
print(my_dict.values())


def myFunc(**kwargs):
    for item in kwargs.items():
        print(item)
myFunc(x=100, y=200, z='b')


a = ((1,2), (3,4), (5,6))
for first, last in a:
    print(first + last)



my_dict = {'a': 1, 'b': 2, 'c': 3}

for key, value in my_dict.items():
    print(f"Key: {key}, Value: {value}")

def test(name, *args):
    print(name)
    print(args)

test('홍길동', 1, 2, 3)

def test(name, **kwargs):
    print(name)
    print(kwargs)

test('홍길동', a=4, b=5, c=6)

def myFunc(*args):
    for arg in args:
        print(arg)
p1 = [10, 20, 'a']
myFunc(p1)
myFunc(*p1)