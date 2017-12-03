def enum(*sequential):
    print(sequential)
    enums = dict(zip(sequential, range(len(sequential))))
    return type('Enum', (), enums)

Numbers = enum('ZERO', 'ONE', 'TWO')
print(Numbers.ZERO)
print(Numbers.ONE)


from enum import Enum, IntEnum   # for enum34, or the stdlib version
# from aenum import Enum  # for the aenum version
key_state = Enum('key_state', 'a b c d')

print(key_state.a)  # returns <Animal.ant: 1>
print(key_state['a'])  # returns <Animal.ant: 1> (string lookup)
print(key_state.a.name)  # returns 'ant' (inverse lookup)


class Animal(Enum):
    ant = 1
    bee = 2
    cat = 3
    dog = 4
print(Animal.ant.name)

e = key_state.a

d = {'a': 1, 'b': 2}
def fun(d):
    d['a'] = 3

fun(d)
print(d)

arrow_key_code = {'left': ord('('), 'right': ord(')'), 'up': ord('+'), 'down': ord('-')}

key = ord('(')
a = key in arrow_key_code.values()
print(arrow_key_code.items())
print(a)
print(arrow_key_code.values())

a = 10
b = 20
c = 30
print(a)
def name(*expr):
    for k in list(locals().values()):
        print(k)

    print("aaa")
    print(list(locals().items()))
    print(list(vars().items()))
    # return [x for x in variables]

name(a, b, c)

import numpy as np

a = [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3,3]]

print(np.nonzero(a))

import cv2
img = cv2.imread('./test_images/straight_lines1.jpg')
print(img.shape)
img = img[:,:,:]
print(img.shape)
new_img = np.zeros_like(img)
new_img[img>190.] = 1
nonzero = np.nonzero(new_img)
print(np.ndim(nonzero))
print(len(nonzero[0]), len(nonzero[1]))
x = nonzero[1]
y = nonzero[0]
good_left = ((x > 100) & (x < 130) & (y > 0) & (y < 110)).nonzero()[0]
print(len(good_left))
print(good_left)
print(x[good_left])
print(y[good_left])

from collections import deque

q = deque(maxlen=10)

q.append(a[0])
q.append(a[1])
print(np.mean(q, axis=0, dtype=np.float32))

q.append(a[2])
print(np.mean(q, axis=0, dtype=np.float32))


l = []
l.append([1,2])
l.append([3,4,5])
e = np.hstack(l)
print(np.mean(e, axis=0, dtype=np.float32))

