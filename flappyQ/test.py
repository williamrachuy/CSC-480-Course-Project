# Hello World program in Python

import math

SCREENWIDTH = 288
XDIVIDE = 30


print(0)
print(SCREENWIDTH / 2 - SCREENWIDTH * 0.2)

dx = range(0, math.floor((SCREENWIDTH / 2 - SCREENWIDTH * 0.2) / XDIVIDE) + 1)
print(dx)

i = 0
print((i - (i % XDIVIDE)) / XDIVIDE)

i = SCREENWIDTH / 2 - SCREENWIDTH * 0.2
print((i - (i % XDIVIDE)) / XDIVIDE)
print(" ")


# ------------------------------------------------------------------
# *** zero is at the top

SCREENHEIGHT = 512
BASEY        = SCREENHEIGHT * 0.79
PIPEGAPSIZE  = 100 # gap between upper and lower part of pipe

YDIVIDE = 30

# high value bird (low on screen):
high_b        = BASEY

# low value bird (top of screen):
low_b = 0

# high value pipe:
high_p = int(BASEY * 0.6 - PIPEGAPSIZE) + PIPEGAPSIZE

# low value pipe:
low_p = 0 + PIPEGAPSIZE


print(PIPEGAPSIZE - BASEY)
print(int(BASEY * 0.6 - PIPEGAPSIZE) + PIPEGAPSIZE)


DYMIN = math.floor((PIPEGAPSIZE - BASEY) / YDIVIDE)
dy = range(DYMIN,
    math.floor((int(BASEY * 0.6 - PIPEGAPSIZE) + PIPEGAPSIZE) / YDIVIDE) + 1)

print(dy)

i = PIPEGAPSIZE - BASEY
print((i - (i % YDIVIDE)) / YDIVIDE)

i = int(BASEY * 0.6 - PIPEGAPSIZE) + PIPEGAPSIZE
print((i - (i % YDIVIDE)) / YDIVIDE)


# ------------------------------------------------------------------

XLEN = len(dx)
YLEN = len(dy)
   
print(" ") 
print(XLEN)
print(YLEN)

v = range(-9, 10)
VLEN = len(v)
# ------------------------------------------------------------------
print(" ") 

DXMIN = 0
VMIN = -9


def map(dx, dy, v):
    dx = (dx - (dx % XDIVIDE)) / XDIVIDE
    dy = (dy - (dy % YDIVIDE)) / YDIVIDE - DYMIN 
    v = v - VMIN
    return int((dx * YLEN * VLEN) + (dy * VLEN) + v)

i = 0;
Qvalues = []

for x in dx:
    for y in dy:
        for z in v:
            Qvalues.append([0,1])


with open('test.txt', 'w') as f:
    for item in Qvalues:
        f.write("(%s,%s) " % (str(item[0]), str(item[1])))

with open('test.txt', 'w') as f:
    for item in Qvalues:
        f.write("(%s,%s) " % (str(item[0]), str(item[1])))

tuples = []
for t in open('test.txt').read().split():
    a,b = t.strip("()").split(',')
    tuples.append([int(a), int(b)])

print(tuples)



print(type(tuples[2]))

'''
count = 0
for x in range(0, math.floor((SCREENWIDTH / 2 - SCREENWIDTH * 0.2)) + 1):
    for y in range(math.floor((PIPEGAPSIZE - BASEY)),
            math.floor((int(BASEY * 0.6 - PIPEGAPSIZE) + PIPEGAPSIZE)) + 1):
        for z in v:
            print(str(x)+", "+str(y)+", "+str(z)+" = "+str(map(x, y, z)))
'''
