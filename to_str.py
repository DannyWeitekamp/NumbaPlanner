import numba 
import numpy as np
import numba
from numba import types, jit,njit,jitclass, guvectorize,vectorize,prange
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.types import ListType, unicode_type
from numba.unicode import _empty_string, _set_code_point, _get_code_point, PY_UNICODE_1BYTE_KIND


DIGITS_START = 48
DOT = 46

import timeit
N=1000
def time_ms(f):
		f() #warm start
		return " %0.6f ms" % (1000.0*(timeit.timeit(f, number=N)/float(N)))


@njit
def int_to_str(x):
	l = 0 
	_x = x
	while _x > 0:
		_x = _x // 10
		l += 1
	s = _empty_string(PY_UNICODE_1BYTE_KIND,l)
	for i in range(l):
		digit = x % 10
		_set_code_point(s,l-i-1,digit + DIGITS_START)
		x = x // 10
	return s

@njit
def float_to_str(x):
	if(x == np.inf):
		return 'inf'
	elif(x == -np.inf):
		return '-inf'
	l1,l2 = 0,-1
	_x = x
	while _x > 0:
		_x = _x // 10
		l1 += 1
	_x = x % 10 
	while _x > 1e-10:
		_x = (_x * 10) % 10
		l2 += 1

	l2 = max(1,l2)
	l = l1+l2+1

	_x = x-np.floor(x)
	s = _empty_string(PY_UNICODE_1BYTE_KIND,l)

	_x = x
	for i in range(l1):
		digit = _x % 10
		_set_code_point(s,l1-i-1,digit + DIGITS_START)
		_x = _x // 10

	# if(l2 > 0):
	_set_code_point(s,l1,DOT)

	_x = x % 10 
	for i in range(l2):
		_x = (_x * 10) % 10
		digit = int(_x)  
		# print('here',i,digit)
		_set_code_point(s,l1+i+1,digit + DIGITS_START)
	return s
	# print(_x)
	# print((x*10)%1)
	# print(np.floor((x*10)%1))
	# while _x > 0:
	# 	_x = _x // 10
	# 	l2 += 1


	# s = _empty_string(PY_UNICODE_1BYTE_KIND,l)
	# for i in range(l):
	# 	digit = x % 10
	# 	_set_code_point(s,l-i-1,digit + DIGITS_START)
	# 	x = x // 10
	# return s

# print(int_to_str(147))
print(float_to_str(146.78))
print(float_to_str(0.78))
print(float_to_str(4.0))
print(float_to_str(1.34e24))

print(time_ms(lambda : float_to_str(146.78)))
