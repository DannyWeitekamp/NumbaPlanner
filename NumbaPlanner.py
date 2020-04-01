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

	compute_it = inner_name+"(%s)" % (",".join(["x%s[i%s]"%(i,i) for i in range(n_args) ]))
	if(dtype == np.unicode):
		prealloc =  "\tout = List.empty_list(unicode_type)\n"
		insert = "out.append(%s)\n" % compute_it
	else:
		prealloc =  "\tout = np.empty((%s),np.dtype(%r))\n" % (",".join(lengths),dtype_name) 
		insert = "out"+ "".join(["[i%s]"%i for i in range(n_args)])+ " = %s\n" % compute_it
			 

	source = header + "def %s(%s):\n"% (func_name,",".join(arg_names))  + \
			 prealloc + \
			 "".join([("\t"*(i+1))+"for i%s in range(len(x%s)):\n" %(i,i)
			 		 for i in range(n_args)]) + \
			 ("\t"*(n_args+1))+ insert + \
			 "\treturn out"
	print(source)
	out = [source] if returnSource else []
	if(returnCompiled):
		f_name = f.__code__.co_name
		_globals = globals()
		_globals.update({inner_name:njit(f,nogil=True,fastmath=True)})
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
	

add_em,_ = gen_sweeps(Add,np.dtype('float'))
cat_em,_ = gen_sweeps(Add,np.dtype('unicode'))


x = np.arange(10)
u = np.array([str(y) for y in x],np.dtype('unicode'))


@njit
def moop():
	u_ = List()
	for x in ['0','1','2','3','4','5','6','7','8','9']:
		u_.append(x)
	# U_ = np.empty(len(u_),dtype=np.dtype('U10'))
	# for i in range(len(u_)):
	# 	U_[i] = u_[i]
	# u_ = ['0','1','2','3','4','5','6','7','8','9']
	# cat_em(U_,U_)
	cat_em(u_,u_)

print("add: ",time_ms(lambda : add_em(x,x)))
# print("cat: ",time_ms(lambda : cat_em(u,u)))
print("cat: ",time_ms(moop))

# print(add_em(x,x))
print(cat_em(u,u))


print("add_em",)