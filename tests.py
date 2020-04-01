import numba 
import numpy as np
import numba
from numba import types, jit,njit,jitclass, guvectorize,vectorize,prange
from numba import deferred_type, optional
from numba import void,b1,u1,u2,u4,u8,i1,i2,i4,i8,f4,f8,c8,c16
from numba.typed import List, Dict
from numba.types import ListType, unicode_type
import timeit
from sklearn import tree as SKTree
from numba.extending import overload

class Operator(object):
	def __init__(self):
		pass
	def condition(self,**args):
		return True

	def forward(self,**args):
		pass

	def backward(self,**args):
		pass

# @jitclass([])
class Add(Operator):
	def __init__(self):
		self.out_type = np.double
	def forward(x,y):
		return x+y
	def backward(g,x):
		return g-x

class Add3(Operator):
	def __init__(self):
		self.out_type = np.double
	def forward(x,y,z):
		return x+y+z
	def backward(g,x,y):
		return g-(x+y)

rule = Add()

N=1000
def time_ms(f):
		f() #warm start
		return " %0.6f ms" % (1000.0*(timeit.timeit(f, number=N)/float(N)))


fA = njit(Add.forward)


# @jit(nogil=True,fastmath=True,cache=True) 
# def apply(f,out,**args):

def gen_sweep(src,f,dtype,returnSource=True,returnCompiled=True):
	n_args = f.__code__.co_argcount
	f_name = f.__code__.co_name
	header = "@jit(nogil=True,fastmath=True)\n"
	lengths = ['len(x%s)' % x for x in range(n_args)]
	arg_names = ['x%s' % x for x in range(n_args)]
	inner_name = "%s_%s"% (src,f_name)
	func_name = "sweep_" + inner_name
	dtype_name = str(dtype) if (isinstance(dtype,np.dtype)) else dtype.__name__
	source = header + "def %s(%s):\n"% (func_name,",".join(arg_names))  + \
			 "\tout = np.empty((%s),np.dtype(%r))\n" % (",".join(lengths),dtype_name) + \
			 "".join([("\t"*(i+1))+"for i%s in range(len(x%s)):\n" %(i,i)
			 		 for i in range(n_args)]) + \
			 ("\t"*(n_args+1))+"out"+ "".join(["[i%s]"%i for i in range(n_args)])+ " = " + \
			 inner_name+"(%s)\n" % (",".join(["x%s[i%s]"%(i,i) for i in range(n_args) ])) + \
			 "\treturn out"
	print(source)
	out = [source] if returnSource else []
	if(returnCompiled):
		f_name = f.__code__.co_name
		_globals = globals()
		_globals.update({inner_name:jit(f)})
		l = {}
		exec(source,_globals,l)
		out.append(l[func_name])

	return out

def gen_sweeps(operator,dtype):
	src = operator.__name__
	out = []
	for f in [operator.forward,operator.backward]:
		source,compiled = gen_sweep(src,f,dtype)
		out.append(compiled)
	
	return out
	

# @jit(nogil=True,fastmath=True,cache=True) 
# def sweep(**args):
	# out = np.empty([len(x) for x in args],np.double)
	# it = np.nditer(tuple([out,**args])):

	# for i in range(len(x)):
	# 	for j in range(len(y)):
			# out[i][j] = fA(x[i],y[j])
	# return out


x_in = np.arange(100,dtype=np.double)
def go():
	return apply_2(x_in,x_in)

# print(time_ms(go))

# f,b = gen_sweeps(Add,np.double)

# x = np.array([1,2,3])

# print(f(x,x))
# def return_add(x, i, y, j):
#     return x[i] + y[j]


add_em,b = gen_sweeps(Add,np.dtype('float'))

x = np.array(['AB','BB','CB','DB','EB'],np.dtype('U2'))
# print(x[0])
# print(x[0] + x[0])
# print(x.dtype)

# print(f(x,x))

@jit(nogil=True,fastmath=True)
def list_str(x0,x1):
	out = List.empty_list(unicode_type)
	for i0 in range(len(x0)):
		for i1 in range(len(x1)):
			out.append(x0[i0]+x1[i1])
	return out

@jit(nopython=True, nogil=True,fastmath=True)
def tensor_str(x0,x1):
	out = np.empty((len(x0),len(x1)),dtype=np.dtype('U10'))
	for i0 in range(len(x0)):
		for i1 in range(len(x1)):
			# print(i0 * len(x1) + i1)
			# print(x0[i0]+x1[i1])
			# out[i0,i1] = x0[i0]+x1[i1]
			# for i in range(len(x0[i0])):
			# 	out[i0,i1][i] = x0[i0][i]
			# out[i0,i1][len(x0[i0]):len(x0[i0])+len(x1[i1])] = x1[i1] 
			out[i0,i1] = (x0[i0]+x1[i1])
	return out


# @jit(nopython=True, nogil=True,fastmath=True)
def tensor_str_python(x0,x1):
	out = []
	for i0 in range(len(x0)):
		out.append([])
		for i1 in range(len(x1)):
			# print(i0 * len(x1) + i1)
			# print(x0[i0]+x1[i1])
			# out[i0,i1] = x0[i0]+x1[i1]
			# for i in range(len(x0[i0])):
			# 	out[i0,i1][i] = x0[i0][i]
			# out[i0,i1][len(x0[i0]):len(x0[i0])+len(x1[i1])] = x1[i1] 
			out[i0].append(x0[i0]+x1[i1])
	return out

# x = np.array([1,2,3],np.dtype('i4'))


# y=list_str(x,x)
print(list_str(x,x))
t=tensor_str(x,x).flatten()
# print(sweep_str(y,y))

# def do_list_str():
# 	list_str(y,y)	

# def do_tensor_str():
# 	tensor_str(y,y)
l = t.tolist()
def do_python():
	tensor_str_python(l,l)	

@njit
def do_list_str2():
	t = List()
	for x in ['AB','BB','CB','DB','EB']:
		t.append(x)
	# t = ['AB','BB','CB','DB','EB']
	list_str(t,t)	

@njit
def do_tensor_str2():
	tensor_str(t,t)	

def do_numpy_str():
	np.char.add(t,t)

# print(time_ms(do_list_str))
# print(time_ms(do_tensor_str))
print("do_python:      ",time_ms(do_python))
print("do_list_str2:   ",time_ms(do_list_str2))
print("do_tensor_str2: ",time_ms(do_tensor_str2))
print("do_numpy_str:   ",time_ms(do_numpy_str))



x_flts = np.array([1,2,3,4,5],np.dtype('float'))
def flt_add():
	add_em(x_flts,x_flts)

# print(time_ms(flt_add))

# print(tensor_str(t,t))
# print(np.char.add(t.reshape(1,-1),t.reshape(-1,1)))


@jit(nopython=True, nogil=True,fastmath=True)
def byte_copy_str(x0,x1):
	out = np.empty((len(x0),len(x1),x0.shape[1]+x1.shape[1]-1),dtype=np.uint8)
	for i0 in range(len(x0)):
		# l0 = 2#np.argmin(x0[i0])
		for i1 in range(len(x1)):
			l0 = 2#np.argmin(x0[i0])
			l1 = 2#np.argmin(x1[i1])
			out[i0,i1][0:l0] = x0[i0][:l0]
			out[i0,i1][l0:l0+l1+1] = x1[i1][:l1+1]

	return out


b_ = np.array([list(b'AB')+[0],
			   list(b'BB')+[0],
			   list(b'CB')+[0],
			   list(b'DB')+[0],
			   list(b'EB')+[0]],np.uint8)

def do_bcs():
	byte_copy_str(b_,b_)

print("do_bcs:         ",time_ms(do_bcs))
print(byte_copy_str(b_,b_))

@overload(str)
def int_to_str(x):
	if(isinstance(x,types.Integer)):
		def to_str(x):
			return '5'
		return to_str


@jit(nopython=True, nogil=True,fastmath=True)
def byte_frm_numbers():
	N = 9
	out = np.empty((N,N,3),dtype=np.uint8)
	for i0 in range(N):
		for i1 in range(N):
			st = (str(i0) + str(i1))
			for k in range(len(st)):
			# print(str(i0)+str(i1))
				out[i0,i1][k] = st[k].data

	return out
# byte_frm_numbers()
# print(byte_copy_str(t.view(np.uint8),t.view(np.uint8)))

# print()
# gen_sweeps(Add3,np.double)



# a = np.array(['a','b','c'])
# @jit(nogil=True,fastmath=True)
# def string_stuff(a):
# 	a+b

# print(string_stuff(a))

# def forward(rules,kb):
# 	for rule in rules:
# 		rule.forward()
	
# def backward(rules,kb):




