#ver.250623_1 Added glide_init argument in collatz_short for correct glide calculation in c_int -> c_short calls in collatz_c.py

import gmpy2 #does not work without this import, although gmp isn't used here o_O (cannot find 'collatz.dll')

import ctypes
import os
from pathlib import Path
from ctypes import POINTER, Structure, c_char_p, c_ubyte, c_int, c_uint, c_ulonglong, c_size_t, c_void_p, c_ulong, byref, addressof, cast
import gc

from timeit import timeit
import pdb


if os.name == 'nt':
 c_present = True

 AUTO_COMPILE = os.getenv('COLLCOMP', '0').lower()
 if '1' in AUTO_COMPILE or 'collatz.dll' not in os.listdir():
  try:
   os.system('gcc -shared -o collatz.dll collatz.c -lgmp -fPIC')
  except Exception:
   c_present = False

 # Load the compiled library
 try:
  lib_path = os.path.abspath('collatz.dll')
  lib = ctypes.CDLL(lib_path)
 except Exception:
  c_present = False

else:
 c_present = False
 
if not c_present:
 print('C library not available, using default functions') 

class IterLog(Structure):
 _fields_ = [
            ('num_buf', POINTER(c_ubyte)),
            ('num_offsets', POINTER(c_size_t)),
            ('buf_capacity', c_size_t),
            ('iter_types', POINTER(c_int)),
            ('ctr', c_size_t),
            ('ctr_lim', c_size_t),
            ('cycle_start', c_size_t),
            ('i_jump', c_size_t),
            ('i_glide', c_size_t),
            ('exit_code', c_int)
]


class CollatzResults(Structure):
 _fields_ = [
            ('n_last_buf', POINTER(c_ubyte)),
            ('n_last_size', c_size_t),
            ('jump_buf', POINTER(c_ubyte)),
            ('jump_size', c_size_t),
            ('ctr', c_size_t),
            ('op_ctrs', POINTER(c_size_t)),
            ('op_ctrs_size', c_size_t),
            ('glide', c_size_t)
]

class CR_Int(Structure):
 _fields_ = [
            ('n_last', c_ulonglong),
            ('jump', c_ulonglong),
            ('ctr', c_size_t),
            ('op_ctrs', POINTER(c_size_t)),
            ('op_ctrs_size', c_size_t),
            ('glide', c_size_t)
]



# Set up argument and return types
lib.collatz.argtypes = [
 POINTER(c_ubyte),              #n_0 bytes,
 c_size_t,                      #len(n_0 bytes)
 c_int,                         #multiplier 
 POINTER(c_int), c_size_t,      #divisors
 c_int,                         #add
 c_int,                         #log_n_max
 c_size_t,                      #ctr_lim
 POINTER(c_int), c_size_t,      #ns_min
 POINTER(IterLog)        #result
] #OK

lib.init_iter_log.argtypes = [POINTER(IterLog), c_int, c_size_t]
lib.free_iter_log.argtypes = [POINTER(IterLog)]
lib.reset_iter_log.argtypes = [POINTER(IterLog), c_size_t]

# Set up argument and return types
lib.collatz_short.argtypes = [
 POINTER(c_ubyte),              #n_0 bytes,
 c_size_t,                      #len(n_0 bytes)
 c_int,                         #multiplier 
 POINTER(c_int), c_size_t,      #divisors
 c_int,                         #add
 c_int,                         #log_n_max
 c_size_t,                      #ctr_lim
 POINTER(c_int), c_size_t,      #ns_min
 c_ulonglong,                   #initial number for glide calculation
 c_ulonglong,                   #n_min_lim: if iterations fall below it, terminate (nonzero in jump-path records calculation
 POINTER(CollatzResults)        #result
] #OK

lib.init_collatz_result.argtypes = [POINTER(CollatzResults), c_size_t]
lib.free_collatz_result.argtypes = [POINTER(CollatzResults)]

lib.free_buffer.argtypes = [POINTER(c_ubyte)]
lib.free_buffer.restype = None

# Set up argument and return types
  #lib.collatz_int(n_int, m, ds, ds_len, a, n_max, lim, nsm, nsm_len, byref(result))
lib.collatz_int.argtypes = [
 c_ulonglong,                         #n_0, 
 c_ulonglong,                         #multiplier 
 POINTER(c_ulonglong), c_size_t,      #divisors, divisors len
 c_ulonglong,                         #add
 c_ulonglong,                         #10**log_n_max
 c_size_t,                            #ctr_lim
 POINTER(c_ulonglong), c_size_t,      #ns_min, number of ns_mins
 POINTER(CR_Int)                      #result
] #OK


lib.init_cr_int.argtypes = [POINTER(CR_Int), c_size_t]
lib.free_cr_int.argtypes = [POINTER(CR_Int)]

def int_to_bytes(n):
 """Convert Python int to big-endian bytes"""
 return n.to_bytes((n.bit_length() + 7) // 8, 'big', signed=False)

def bytes_to_int(b):
 """Convert bytes back to Python int"""
 return int.from_bytes(b, 'big', signed=False)
 
def iter_log_init(log_n_max=200, ctr_lim=100500):
 iter_log = IterLog()
 lib.init_iter_log(byref(iter_log), c_int(log_n_max), c_size_t(ctr_lim))
 return iter_log

def collatz_c_full(n_0, mult=3, divisors=[2,], add=1, log_n_max=200, ctr_lim=100500,
                   ns_min=[1,], num_buf=None, log_slice=1, ftype=''):
 
 ctr_lim = int(ctr_lim)
 #precalculating
 len_divs, len_ns = len(divisors), len(ns_min)
 
 #convert inputs to ctypes
 n_bytes = int_to_bytes(n_0)
 len_bytes = c_size_t(len(n_bytes))
 n_buf = (c_ubyte * len(n_bytes))(*n_bytes)
 m = c_int(mult)
 ds = (c_int * len_divs)(*divisors)
 ds_len = c_size_t(len_divs)
 a = c_int(add)
 lnm = c_int(log_n_max)
 lim = c_size_t(ctr_lim)   
 nsm = (c_int * len_ns)(*ns_min)
 nsm_len = c_size_t(len_ns)

 if num_buf is None: #or sizes differ
  result = iter_log_init(log_n_max, ctr_lim)

 else:
  ctr_lim_adj = max(ctr_lim, 1000)
  result = num_buf 
  prev_cap = result.buf_capacity
  needed_cap = int( (( ((log_n_max * 3.33) + 64) * ctr_lim_adj) + 7) / 8 ) 
  if needed_cap < prev_cap:
   lib.reset_iter_log(byref(result), c_size_t(ctr_lim_adj))
  else:
   prev_buf_size = lib.free_iter_log(byref(result))
   result = iter_log_init(log_n_max, ctr_lim_adj)
  
 try:
  lib.collatz(n_buf, len_bytes, m, ds, ds_len, a, lnm, lim, nsm, nsm_len, byref(result))
  ec = result.exit_code
  if ec < 0:
   print('some error on c-side')
   lib.free_iter_log(byref(result))
   return [], [], ec, None
  
  il = [] #[0]*(result.ctr+1)
 
  combined_buf = cast(result.num_buf, POINTER(c_ubyte * result.buf_capacity)).contents
  offsets = result.num_offsets[:result.ctr + 2]
  glide = result.i_glide
  i_jump = result.i_jump
  ctr = result.ctr
  end_cond = result.exit_code
  ol = [int(result.iter_types[i]) for i in range(0, result.ctr+1, log_slice)]

  op_types = (0,) + tuple(sorted(divisors))
  op_ctrs = [0 for x in op_types]
  for i in range(len(op_types)):
   op_ctrs[i] = sum([x == op_types[i] for x in ol[:-1]])

  n_last_l, n_last_r = offsets[ctr], offsets[ctr+1]
  num_bytes_fin = bytes(combined_buf[n_last_l:n_last_r])
  n_last = int.from_bytes(num_bytes_fin, 'big')

  jump_l, jump_r = offsets[i_jump], offsets[i_jump+1]
  num_bytes_jump = bytes(combined_buf[jump_l:jump_r])
  jump = int.from_bytes(num_bytes_jump, 'big')
  
  if 'short' not in ftype:   
   for i in range(0, result.ctr+1, log_slice):
    n_start, n_end = offsets[i], offsets[i+1]
    if n_end > n_start:
     num_bytes = bytes(combined_buf[n_start:n_end])
     il.append(int.from_bytes(num_bytes, 'big'))
  
 finally:
  if 'free' in ftype:
   lib.free_iter_log(byref(result))
   result = None
   
 return (ctr, op_ctrs, n_0, n_last, jump, glide, end_cond), il, ol, result


def collatz_c_short(n_0, mult=3, divisors=[2,], add=1, log_n_max=200, ctr_lim=100500, ns_min=[1,], n_min_lim=0, glide_init=0, ftype=''):
 ctr_lim = int(ctr_lim)
 #precalculating
 len_divs, len_ns = len(divisors), len(ns_min)
 
 #convert inputs to ctypes
 n_bytes = int_to_bytes(n_0)
 len_bytes = c_size_t(len(n_bytes))
 n_buf = (c_ubyte * len(n_bytes))(*n_bytes)
 m = c_int(mult)
 ds = (c_int * len_divs)(*divisors)
 ds_len = c_size_t(len_divs)
 a = c_int(add)
 lnm = c_int(log_n_max)
 lim = c_size_t(ctr_lim)   
 nsm = (c_int * len_ns)(*ns_min)
 nsm_len = c_size_t(len_ns)
 gl_init = c_ulonglong(glide_init)
 nminlim = c_ulonglong(n_min_lim)
 result = CollatzResults()
 n_divs = c_size_t(len(divisors))
 lib.init_collatz_result(byref(result), n_divs)
 
 op_ctrs = [0 for i in range(1 + len(divisors))]
 try:
  lib.collatz_short(n_buf, len_bytes, m, ds, ds_len, a, lnm, lim, nsm, nsm_len, gl_init, nminlim, byref(result))
  n_last = bytes_to_int(bytes(cast(result.n_last_buf, POINTER(c_ubyte * result.n_last_size)).contents))
  jump = bytes_to_int(bytes(cast(result.jump_buf, POINTER(c_ubyte * result.jump_size)).contents))
  ctr = result.ctr
  for i in range(len(divisors)+1):
   op_ctrs[i] = result.op_ctrs[i]
  glide = result.glide
 
 finally:
  lib.free_collatz_result(byref(result))

#  if result.n_last_buf:
#   lib.free_buffer(result.n_last_buf)
#  if result.jump_buf:
#   lib.free_buffer(result.jump_buf)

 end_cond = -1
 if n_last < n_min_lim:
  end_cond = 5
 elif n_last > 10**log_n_max:
  end_cond = 2 #likely diverged
 elif ctr >= ctr_lim:
  end_cond = 3
 elif n_last == n_0:
  end_cond = 4
 elif n_last in ns_min:
  end_cond = 0 #converged to a known cycle
 elif n_last not in ns_min and ctr < ctr_lim and n_last < 10**log_n_max:
  end_cond = 1 #new cycle found
  
 #if a new cycle is found, stats are determined incorrectly (some of iterations may continue past first cycle_n_min) 
 #in this case, call with new n_min to calculate stats correctly
 if end_cond == 1 and 'once' not in ftype:
  vs_int = collatz_c_short(n_0, mult=mult, add=add, ctr_lim=ctr_lim, log_n_max=log_n_max, ns_min=list(ns_min) + [n_last,], divisors=divisors, glide_init = glide_init, ftype=ftype + ' once')
  ctr, op_ctrs, n_0, n_last, jump, glide, end_cond = vs_int
  end_cond = 1
  return ctr, op_ctrs, n_0, n_last, jump, glide, end_cond

 return ctr, op_ctrs, n_0, n_last, jump, glide, end_cond


def collatz_c_int(n_0, mult=3, divisors=[2,], add=1, ctr_lim=100500, log_n_max=18, ns_min=[1,], ftype=''):
 ctr_lim = int(ctr_lim)
 #precalculating
 len_divs, len_ns = len(divisors), len(ns_min)
 n_lim = min(int(10**log_n_max), int(2**63 / mult))
 
 #convert inputs to ctypes
 n_int = c_ulonglong(n_0)
 m = c_ulonglong(mult)
 ds = (c_ulonglong * len_divs)(*divisors)
 ds_len = c_size_t(len_divs)
 a = c_ulonglong(add)
 n_max = c_ulonglong(n_lim)
 lim = c_size_t(ctr_lim)
 nsm = (c_ulonglong * len_ns)(*ns_min)
 nsm_len = c_size_t(len_ns)
 result = CR_Int()
 n_divs = c_size_t(len(divisors))
 lib.init_cr_int(byref(result), n_divs)
 
 try: #lib.collatz_short(n_buf, len_bytes, m, ds, ds_len, a, lnm, lim, nsm, nsm_len, byref(result))
  lib.collatz_int(n_int, m, ds, ds_len, a, n_max, lim, nsm, nsm_len, byref(result))
  n_last = result.n_last
  jump = result.jump
  ctr = result.ctr
  op_ctrs = [0 for i in range(1 + len(divisors))]
  for i in range(len(divisors)+1):
   op_ctrs[i] = result.op_ctrs[i]
  glide = result.glide
 
 finally:
  lib.free_cr_int(byref(result))
 
 end_cond = -1
 if n_last > n_lim:
  end_cond = 2 #overflow
 elif ctr == ctr_lim:
  end_cond = 3
 elif n_last == n_0:
  end_cond = 4
 elif n_last in ns_min:
  end_cond = 0 #converged to a known cycle
 elif n_last not in ns_min and ctr < ctr_lim and n_last < 2**62:
  end_cond = 1 #new cycle found
  
 #if a new cycle is found, stats are determined incorrectly (some of iterations may continue past first cycle_n_min) 
 #in this case, call with new n_min to calculate stats correctly
 if end_cond == 1 and 'once' not in ftype:
  vs_int = collatz_c_int(n_0, mult=mult, add=add, ctr_lim=ctr_lim, log_n_max=log_n_max, ns_min=list(ns_min) + [n_last,], divisors=divisors, ftype=ftype + ' once')
  ctr, op_ctrs, n_0, n_last, jump, glide, end_cond = vs_int
  end_cond = 1
  return ctr, op_ctrs, n_0, n_last, jump, glide, end_cond
  
 return ctr, op_ctrs, n_0, n_last, jump, glide, end_cond
