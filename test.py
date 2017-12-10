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

a = [[1, 2, 3, 2, 1], [2, 3, 4, 3, 2], [3, 4, 5, 4,3], [2,4,6]]

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
q1 = deque(maxlen=10)
q1.append(a[3])
q.append(a[0])
q.append(a[1])
print(np.mean(q, axis=0, dtype=np.float32))

q.append(a[2])
print(np.mean(q, axis=0, dtype=np.float32))

s1 = np.array(q)
s2 = np.array([2,4,5])
s2 = s2.reshape((-1, 1))
print('s1', s1)
print('s2', s2)
s3 = s1*s2
print(s3)
m1 = np.sum(s3, axis=0)
m2 = np.sum(s2)
print('----')
print(m1)
print(m2)
print(m1/m2)
print('------')

print(np.mean(s3, axis=0, dtype=np.float32))



l = []
l.append([1,2])
l.append([3,4,5])
e = np.hstack(l)
print(np.mean(e, axis=0, dtype=np.float32))

a = np.array([])
a = []
b = np.array([1, 2, 3])
b = [1,2,3]
c = None
print (a is None)
print (a != None)
print (b is None)
print (b != None )
print(b == [])
print(a == [])
print (a == b)
print(c is None)
print(c == [])
b = np.zeros(5)
print(b)
c = [1,2,3]
print(np.append(c, b))
a = np.array([[1,2,3]])
b = np.zeros([3, 3])
print(a)
print(b)
print(np.append(a, b, axis=0))
print(a)

a = np.arange(12).reshape(3,4)
print(a)
print(a[:,:-1])

q = deque(maxlen=10)
for i in range(15):
    q.append(i)
print(q)

a = [[1, 2, 3, 2, 1], [2, 3, 4, 3, 2], [3, 4, 5, 4,3], [2,4,6]]
print(np.concatenate(a, axis=0))