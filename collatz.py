from collatz_array import *
from collatz_binary import *
from collatz_complex import *
from collatz_random import *
from collatz_c import *

#see 'collatz' function docstring for basic descriptions

'''
import sys
dir_code = 'D:\\Code\\math_custom'
if dir_code not in sys.path:
 sys.path.insert(0, dir_code) 

from math_custom import *
'''  

#------------------------------------------

@njit
def collatz_iter(n, mult=3, divisors=div_def, add=1):
 '''
 Single collatz iteration for numba-accelerated calculation
 '''
 for div in divisors:
  res, rem = divmod(n, div)
  if rem == 0: #even step
   return res, div
 return n*mult + add, 0 #odd step

def collatz_iter_def(n, mult=3, divisors=div_def, add=1):
 '''
 Single collatz iteration for arbitrarily large numbers
 '''
 for div in divisors:
  res, rem = divmod(n, div)
  if rem == 0: #even step
   return res, div
 return n*mult + add, 0 #odd step


def collatz_def(n_0, mult=3, add=1, divisors = div_def, ctr_lim=100500, log_n_max=80, 
               ns_min=ns_min_def, il_prev=[], ol_prev=[], ftype=''): #basic function for extended nx/[divs]+add sequence calculation

 '''
 Collatz sequence iteration and operation type log calculation for arbitrarily large numbers.
 For full description of arguments and return values, see collatz_calc docstring.
 Specific arguments: il_prev, ol_prev - iteration and iteration type logs appended to the beginning of current logs.
 
 '''
 ctr_lim = int(ctr_lim)
 ctr_cubes = [i**3 for i in range(2, max(ceil(ctr_lim**(1/3)) + 2, 4))]
 
 iter_log, odd_log = il_prev + [n_0,], ol_prev + []
 cond, ctr, n, end_cond = True, len(iter_log), n_0, -1
 ctr_check = ceil(ctr**(1/3))
 
 max_fin = max(ns_min)
 
 n_max = 10**log_n_max
 n_start = iter_log[0] #in case of imported log
 new_cycle_found = False

 while cond:
  n, step_type = collatz_iter_def(n, mult=mult, divisors=divisors, add=add)
  iter_log.append(int(n))
  odd_log.append(step_type)
       
  end_cond = -1
  if n == n_start: #cycle detected; #if starting number belongs to a cycle and no ns_min are specified: return full cycle
   end_cond = 4
   cond = False

  if n > n_max: #divergence or overflow detection
   end_cond = 2
   cond = False
   
  if ctr >= ctr_lim:
   end_cond = 3
   cond = False

  if cond and n <= max_fin:
   for n_m in ns_min:
    if n == n_m:
     cond = False
     end_cond = 0 #converged to a known cycle
     break    

  #to return full cycles if needed, set ns_min == []
  if end_cond != 4 and ctr > 8 and (ctr == ctr_cubes[ctr_check] or ctr == ctr_lim): #** (1/3) % 1 == 0 #frac pwr really slows down! 
   i_prev = 0 if ctr == ctr_lim or sqrt(ctr_check) % 1 == 0 else ctr_cubes[ctr_check-1] #some times, check full log because at slopes ~ 0, cycles can be really large (see 23*n/[2,3,7,13,19]+1 example on 629
   ctr_check += 1
   n_min, ccl = cycle_det_def(iter_log[i_prev:])
   if n_min != 0:
    i_min = iter_log.index(n_min)
    iter_log, odd_log = iter_log[:i_min+1], odd_log[:i_min+1]
    end_cond = 1 #converged to previously unknown cycle
    new_cycle_found = True
    cond = False
    
  ctr += 1
 
 if not new_cycle_found:
  odd_log.append(collatz_iter_def(n, mult=mult, divisors=divisors, add=add)[1])
     
 return iter_log, odd_log, end_cond


@njit
def collatz_jit(n_0, mult=3, add=1, ns_min=ns_min_def, divisors=div_def, 
                ctr_lim=100500, log_n_max=18, ftype=''): 
 '''
 The same as collatz_def, but accelerated by Numba; accepts and processes numbers < long long int
 See collatz_calc docstring for full args/return description
 '''

 n_max = int(10**min(log_n_max, 18))

 ctr_cubes = [i**3 for i in range(2, max(ceil(ctr_lim**(1/3)) + 1, 4))]
 ctr_check = 1
 cond, ctr, n, end_cond = True, 0, n_0, -1

 iter_log, odd_log = [n_0,], [0,] #initializing iteration log and iteration type log (even - 0, odd - 1)
 p1 = odd_log.pop(-1)

 max_fin = max(ns_min)
 
 new_cycle_found = False
 
 while cond:
  n, step_type = collatz_iter(n, mult=mult, divisors=divisors, add=add)
  iter_log.append(n)
  odd_log.append(step_type)

  if n == n_0: #cycle detected; #if starting number belongs to a cycle and no ns_min are specified: return full cycle
   end_cond = 4
   cond = False

  if n > n_max: #divergence or overflow detection
   end_cond = 2
   cond = False

  if ctr >= ctr_lim - 1: #adjusted for iter_log already containing n_0
   end_cond = 3
   cond = False
   
  if cond and n <= max_fin:
   for n_m in ns_min:
    if n == n_m:
     cond = False
     end_cond = 0 #converged to a known cycle
     break    

  #to return full cycles if needed, set ns_min == []
  if end_cond != 4 and ctr > 8 and (ctr == ctr_cubes[ctr_check] or ctr == ctr_lim): #** (1/3) % 1 == 0 #frac pwr really slows down! 
   i_prev = ctr_cubes[ctr_check-1]
   ctr_check += 1
   n_min, ccl = cycle_det(iter_log[i_prev:])
   if n_min != 0:
    i_min = iter_log.index(n_min)
    iter_log, odd_log = iter_log[:i_min+1], odd_log[:i_min+1]
    end_cond = 1 #converged to previously unknown cycle
    new_cycle_found = True
    cond = False
    
  ctr += 1

 if not new_cycle_found:
  odd_log.append(collatz_iter(n, mult=mult, divisors=divisors, add=add)[1])
    
 return iter_log, odd_log, end_cond


#do not calculate log; return only n_last and counters
#return n_last, jump, ctr, ctr_odd, ctr_even, glide
def collatz_short(n_start, mult=3, add=1, ctr_lim=100500, log_n_max=80,
               ns_min=[1,], divisors = div_def, log_slice=0, ftype=''): #basic function for extended nx/[divs]+add sequence calculation
 '''
 Collatz sequence for arbitrarily large numbers, without full log writing.
 Python version of collatz_c_short for tests and comparison.
 See collatz_calc docstring for full arguments/return description.
 '''
 ctr_lim = int(ctr_lim)
 nsm_set = set(ns_min)
 max_fin = max(ns_min) 
 n_max = 10**log_n_max

 cond = True
 ctr, glide, jump, glide_det = 0, 0, n_start, True 
 op_types = (0,) + tuple(sorted(divisors))
 op_ctrs = [0 for i in range(1 + len(divisors))]
 n = n_start
 
 ctr_check, cycle_found, n_check, n_min = 10, False, n, n
 end_cond = -1
 
 iter_log, odd_log = [], []
 while cond:
  if log_slice > 0 and (ctr % log_slice == 0):
   iter_log.append(int(n))

  n, step_type = collatz_iter_def(n, mult=mult, divisors=divisors, add=add)

  if log_slice > 0 and (ctr % log_slice == 0):
   odd_log.append(step_type)

  op_ctrs[op_types.index(step_type)] += 1

  ctr += 1
  
  if glide_det and n < n_start:
   glide_det = False
   glide = ctr
  
  if n > jump:
   jump = n

  if n == n_start: #cycle detected; #if starting number belongs to a cycle and no ns_min are specified: return full cycle
   end_cond = 4
   cond = False
  elif n > n_max: #divergence or overflow detection
   end_cond = 2
   cond = False
   
  if ctr >= ctr_lim:
   cond = False
   end_cond = 3

  if cond and n <= max_fin:
   if n in nsm_set:
     cond = False
     end_cond = 0 #converged to a known cycle
 
  #new cycle detection
  if cond:
   if cycle_found:
    if n < n_min:
     n_min = n
    elif n == n_min:
     cond = False
     end_cond = 1
   if not cycle_found and n == n_check:
    cycle_found = True
    n_min = n
   if not cycle_found and ctr == ctr_check:
    n_check = n
    ctr_check = int(ctr_check * 3 / 2)
 
 if glide_det:
  glide = ctr
 
 if log_slice > 0 and iter_log[-1] != n:
  iter_log.append(n)
  odd_log.append(collatz_iter_def(n, mult=mult, divisors=divisors, add=add)[1])
  
 if end_cond == 1 and 'once' not in ftype:
  vs_int, iter_log, odd_log = collatz_short(n_start, mult=mult, add=add, ctr_lim=ctr_lim, log_n_max=log_n_max, ns_min=list(ns_min) + [n_min,], divisors=divisors, log_slice=log_slice, ftype=ftype + ' once')
  ctr, op_ctrs, n_start, n_last, jump, glide, end_cond = vs_int
  end_cond = 1
  return (ctr, op_ctrs, n_start, n_last, jump, glide, end_cond), iter_log, odd_log
  
 return (ctr, op_ctrs, n_start, n, jump, glide, end_cond), iter_log, odd_log


#------------------------------------------

def collatz(n_0, mad = None, mult=3, add=1,  divisors=div_def, ctr_lim=100500, log_n_max=1000, jit_thr=17, 
            num_buf=None, ns_min=ns_min_def, ns_min_dict=None, log_slice=0, ftype=''):
 '''
 Wrapper/return unification function for basic collatz sequence calculation functions, according to starting parameters and ftype. 

 Calculation procedure. 
 Generalized Collatz function is defined as follows.
 If number is not divisible by any from the set of prime 'divisors', multiply it by 'mult' and add 'add'. 
 Otherwise, divide it by all primes included in divisors until it is no longer divisible by any of them.
 Repeat iterations until any of the following end conditions:
 -An end value from 'ns_min' is reached (typically, smallest members of known cycles) (0)
 -A new closed cycle has been detected (1)
 -Number exceeds 10**log_n_max (2)
 -Iteration number limit is reached (3)
 If no limits were hit: if number belonged to a cycle, return full cycle (4); otherwise - stop at cycle minimum.

 Additional arguments:
 jit_thr - maximum number processed by functions using long long int type (collatz_jit and collatz_c_int). If exceeded, switches to arbitrary-precision functions.
 num_buf - reusable bytes-type buffer for C-based collatz_c_full function
 ns_min_dict - dictionary of all end values (with keys as ns_min and values containing different stats; used in memorizing records calculation)

 Returns:
 -integer statistics: counts of all, odd (multiplying) and even (dividing) iterations, last number, max reached number ('jump'), number of iterations until number falls below starting, end condition. Always calculated.
 -iteration log: list of all resulting numbers from starting to ending, ([] if not calculated)
 -operation log ('odd log'): list of iteration types: 0 - 'odd', 2, 3, ... - divided by 2, 3, ... ([] if not calculated)
 -result buffer used or generated by collatz_c_full (None in all other types)
 Iteration and op type logs are sliced by log_slice arguments.

 ftype codes: 
 'c_int', 'short', 'py_short': fast calculation without full log writing and return (c long long int, c mpz_t, python versions)
 'jit', 'c_full', 'def' - calculate full logs (jitted long long int, c mpz_t, python versions)
 otherwise: start with 'jit' function and switch to 'def' if overflow is detected.
 
 'ext' - include all primes up to mult in divisors
 
'''

 if not c_present:
  ftype = ftype.replace('short', 'py_short').replace('c_int', 'py_short').replace('c_full', 'def')
  
 if isinstance(mad, tuple) and len(mad) == 3:
  mult, add, divisors = mad

 ctr_lim = int(ctr_lim)
 f_types = ftype.split(' ')
  
 if 'ext' in f_types:
  divisors = [x for x in primes[:mult] if x < mult] #crude

 if 'short' in f_types or 'c_int' in f_types and log10(n_0) > jit_thr:
  vs_int = collatz_c_short(n_0, mult=mult, divisors=divisors, add=add, ctr_lim=ctr_lim, log_n_max=log_n_max, ns_min=ns_min)
  return vs_int, [], [], None

 if 'c_int' in f_types:
  vs_1 = collatz_c_int(n_0, mult=mult, divisors=divisors, add=add, log_n_max = jit_thr, ctr_lim=ctr_lim, ns_min=ns_min)
  if vs_1[-1] != 2:
   return vs_1, [], [], None
  else: #overflow detected: recalculate with mpz version
   n_1 = vs_1[i_n_last]
   vs_2 = collatz_c_short(n_1, mult=mult, divisors=divisors, add=add, ctr_lim=ctr_lim, log_n_max=log_n_max, glide_init = n_0, ns_min=ns_min)
   ctr, n_last, end_cond = vs_1[i_ctr] + vs_2[i_ctr], vs_2[i_n_last], vs_2[i_end_cond]
   op_ctrs = [vs_1[i_ctrs][i] + vs_2[i_ctrs][i] for i in range(len(divisors)+1)]
   jump = max(vs_1[i_jump], vs_2[i_jump])
   glide = vs_1[i_glide] if vs_1[i_glide] < vs_1[i_ctr] else vs_1[i_glide] + vs_2[i_glide] #correct this later (add n_0_init argument in c_short; could be only c_ulonglong because outputted from collatz_int)
   return (ctr, op_ctrs, n_0, n_last, jump, glide, end_cond), [], [], None
   
 if 'py_short' in f_types:
  vs_int, iter_log, odd_log = collatz_short(n_0, mult=mult, divisors=divisors, add=add, ctr_lim=ctr_lim, log_n_max=log_n_max, ns_min=ns_min, log_slice=log_slice)
  return vs_int, iter_log, odd_log, None
   
 if 'c_full' in f_types:
  return collatz_c_full(n_0, mult=mult, divisors=divisors, add=add, ctr_lim=ctr_lim, log_n_max=log_n_max, ns_min=ns_min, 
         num_buf=num_buf, log_slice=log_slice, ftype=ftype) #if 'free' in ftypes: delete buffer
 
 if 'def' in f_types or isinstance(ns_min_dict, dict):
  iter_log, odd_log, end_cond = collatz_def(n_0, mult=mult, add=add, ctr_lim=ctr_lim, log_n_max=log_n_max, ns_min=ns_min, divisors=divisors, ftype='def')

 else: #adaptive calculation: start with the fastest jitted function, switch to default version if number approaches int64 limit
  if n_0 < 10**jit_thr:
   iter_log, odd_log, end_cond = collatz_jit(n_0, mult=mult, add=add, ctr_lim=ctr_lim, log_n_max=min(log_n_max, jit_thr-2), 
                                 ns_min=ns_min, divisors=divisors, ftype=ftype)
   if end_cond == 2 and log_n_max > jit_thr: #close-to-overflow detected in collatz_jit
    iter_log, odd_log, end_cond = collatz_def(iter_log[-1], mult=mult, add=add, ctr_lim=ctr_lim, log_n_max=log_n_max, 
                                  il_prev = iter_log[:-1], ol_prev = odd_log[:-1], ns_min=ns_min, divisors=divisors, ftype='') 
  else:
   iter_log, odd_log, end_cond = collatz_def(n_0, mult=mult, add=add, ctr_lim=ctr_lim, log_n_max=log_n_max, ns_min=ns_min, divisors=divisors, ftype='adapt') 
 
 
 add_vals = ns_min_dict[iter_log[-1]] if isinstance(ns_min_dict, dict) and iter_log[-1] in ns_min else ()
 ctr, op_ctrs, n_0, n_last, jump, glide = log_stats_int(iter_log, odd_log, add_vals=add_vals, mad=(mult, add, divisors))
   
 return (ctr, op_ctrs, n_0, n_last, jump, glide, end_cond), iter_log, odd_log, None
  
#----------------aux functions------------------
   

def cycle_det_def(n_log):
 '''
 New cycle detection
 Takes iteration log, returns last occurence of cycle minimum number and full cycle (otherwise dummy values)
 '''
 len_log = len(n_log)
 n_last = n_log[-1]
 cycle_found = False
 for i_n in range(len_log-2, -1, -1):
  if n_log[i_n] == n_last:
   cycle_found = True
   break
 
 if not cycle_found:
  return 0, [0,] #default cycle which always exists
  
 cycle_raw = n_log[i_n:len_log-1]
 cycle_abs = [abs(n) for n in cycle_raw]
 i_min = cycle_abs.index(min(cycle_abs))
 cycle_min = cycle_raw[i_min]
 ccl = cycle_raw[i_min:] + cycle_raw[:i_min]
 
 return cycle_min, ccl


@njit
def cycle_det(n_log):
 '''
 Cycle detection in iteration log
 returns first occurence of cycle minimum number and full cycle if cycle present, otherwise dummy values
 '''
 len_log = len(n_log)
 n_last = n_log[-1]
 cycle_found = False
 for i_n in range(len_log-2, -1, -1):
  if n_log[i_n] == n_last:
   cycle_found = True
   break
 
 if not cycle_found:
  return 0, [0,] #default cycle which always exists
  
 cycle_raw = n_log[i_n:len_log-1]
 cycle_abs = [abs(n) for n in cycle_raw]
 i_min = cycle_abs.index(min(cycle_abs))
 cycle_min = cycle_raw[i_min]
 ccl = cycle_raw[i_min:] + cycle_raw[:i_min]
 
 return cycle_min, ccl



#----------------------------stats--------------------------------------


#  min_val = n if n < min_val else n_0

def log_stats_int(iter_log, odd_log, divisors=(), mad=None, add_vals=()): #OK
 '''
 calculates integer statistics for collatz iteration log, see 'Integer statistics' in collatz docstring
 '''
 if isinstance(mad, tuple):
  divisors = mad[2]
  
 op_types = (0,) + tuple(sorted(divisors)) if len(divisors) > 0 else tuple(int(x) for x in np.unique(odd_log))
 added_vals = len(add_vals) == 3
 op_ctrs_fin, jump_fin, n_last = add_vals if added_vals else ([0 for i in range(len(op_types))], 1, iter_log[-1])
 ctr_fin = sum(op_ctrs_fin)
 
 n_start = iter_log[0] 
 jump = max(max(iter_log), jump_fin)
 
 op_ctrs = [0 for x in op_types]
 for i in range(len(op_types)):
  op_ctrs[i] = sum([x == op_types[i] for x in odd_log[:-1]]) + op_ctrs_fin[i]

 ctr = sum(op_ctrs)
 
 glide=ctr
 for i in range(len(iter_log)):
  if iter_log[i] < n_start:
   glide=i
   break
  
 return ctr, op_ctrs, n_start, n_last, jump, glide

# gamma = ctr_even / log(abs(n_start)) if abs(n_start) != 1 else 0.0
# res_exp = ctr_even * log(max(divisors)) - ctr_odd * log(mult) - log(n_start) #check #prod( [(1 + 1 / (mult*abs(n))) for n in odd_log] )
# res_exp = res_exp if abs(res_exp) > 100 else exp(res_exp)


def log_stats_float(n_start, n_last, op_ctrs, jump):
 '''
 Calculates float-type statistics for collatz sequence, see 'Integer statistics' in collatz docstring for arg description)
 Returns:
 slope = (log10(n_last) - log10(n_start)) / (total iteration count): measure of average change in single iteration
 completeness = ratio of numbers of odd and even iterations 
 log_jump - ratio of logarithms of maximum reached number and the starting number 
 '''
 ctr = sum(op_ctrs)
 ctr_odd = op_ctrs[0]
 ctr_even = ctr - ctr_odd
 compl = ctr_odd / ctr_even if ctr_even != 0 else 0.0 #last number is not included, but by construction, it it always odd.
 slope = (log10(n_last) - log10(n_start)) / ctr if ctr > 0 else nan
 log_jump = log(jump) / log(n_start) if abs(n_start) != 1 else log(jump)
 return slope, compl, log_jump

def log_stats_full(iter_log, odd_log, n_range=(1, 100000), add_vals=()):
 '''
 Calculates integer-type statistics for every number in iteration log which falls in n_range limits
 Returns dictionary of integer-type statistice, see collatz docstring.
 Very slow, not used.
 '''
 ctr_o_add, ctr_e_add, jump_add, n_last = add_vals if len(add_vals) == 4 else (0, 0, 0, il[-1])
 range_min = int(n_range[0])
 range_max = int(n_range[1])
 
 n_start, n_min = iter_log[0], min(iter_log)
 is_cycle = n_last == n_start
 i_start = 1 if is_cycle and n_start == n_min else 0
 i_end = len(iter_log) if not is_cycle or i_start == 1 else iter_log.index(n_min) + 1
 il = iter_log[i_start:i_end]
 ol = odd_log[i_start:i_end]
  
 stats_dict = {}
 
 is_odd = [1 if x==0 else 0 for x in ol[:-1]] + [0,]
 is_even = [0 if x==0 else 1 for x in ol[:-1]] + [0,]
 ctrs = np.arange(len(il)-1, -1, -1)
 ctrs_odd = np.cumsum(is_odd[::-1])[::-1] + ctr_o_add
 ctrs_even = np.cumsum(is_even[::-1])[::-1] + ctr_e_add
 jumps = []
 n_max = max(il[-1], jump_add)
 for i in range(len(il)-1, -1, -1):
  if il[i] > n_max:
   n_max = il[i]
  jumps.append(n_max)
 jumps.reverse()

 for i in range(len(il)): #exclude last member of a cycle
  n_cur = il[i]
  if range_min <= n_cur <= range_max:
   stats_dict[n_cur] = (int(ctrs_odd[i]), int(ctrs_even[i]), jumps[i], n_last)
    
 return stats_dict
 

def collatz_calc(n_0, mad=None, mult=3, add=1, divisors=div_def, ctr_lim=100500, log_n_max=80, jit_thr=16, ns_min=ns_min_def, 
                 ns_min_dict = None, log_slice=0, ftype=''):
 '''
 Same as collatz but returns integer stats, float stats, iter log, operation type log and result buffer (or dummies for last three values) 
 See collatz and log_stats_float docstrings
 ''' 
 if isinstance(mad, tuple) and len(mad) == 3:
  mult, add, divisors = mad

 ctr_lim = int(ctr_lim)
 vs_int, iter_log, odd_log, res_buf = collatz(n_0, mult=mult, divisors=divisors, add=add, ctr_lim=ctr_lim, ns_min=ns_min, ns_min_dict = ns_min_dict,
              log_n_max=log_n_max, jit_thr=jit_thr, log_slice=log_slice, ftype=ftype)

 ctr, op_ctrs, n_0, n_last, jump, glide, end_cond = vs_int
 vs_float = log_stats_float(n_0, n_last, op_ctrs, jump)
 return vs_int, vs_float, iter_log, odd_log, res_buf

#end_cond: 0: converged to a cycle, 1: n_0 belongs to a cycle, 2: hit 10**log_n_max (likely diverged), -1: hit ctr limit without convergence or divergence

def collatz_speed_calc(mad, ftype='short', n_digits=(100,200,500,1000,2000,5000), t_single=0.03):
 '''
 Calculates number of single iterations per second for Collatz sequences defined by (mult, add, divisors) and calculated by collatz_calc function with ftype 'ftype'
 
 Takes:
 mad: (mult, add, divisors), ftype: type of collatz_calc functions, numbers of digits in the starting number, single test calculation time
 
 Returns:
 dictionary {k:v} where k is ftype(s), v is numpy array of n_digits vs. number of iterations per second

 '''

 
 if isinstance(ftype, tuple) or isinstance(ftype, list):
  vs = {}
  for k in ftype:
   vs.update(collatz_speed_calc(mad, ftype=k, n_digits=n_digits, t_single=t_single))
  return vs
  
 n_0 = range_gen((1000, 1001), 1, [])[0] 
 slope_avg = collatz_calc(n_0, mad=mad, ctr_lim=1e4, log_n_max=2000, ftype=ftype)[i_float][i_slope]

 nd_arr = np.array(n_digits).astype(float)
 nips_arr = np.zeros_like(nd_arr)
 n_nds = nd_arr.size

 for i_nd in range(n_nds):
  nd = n_digits[i_nd]
  calc_len = min(1e4, abs(nd / (3*slope_avg)))
  ns = range_gen((nd, nd+1), 20)
  t_0 = time.time()
  t_el = 0.0
  ctr_tot = 0
  while t_el < t_single:
   n = ns[random.randint(0, 19)]
   vs = collatz_calc(n, mad=mad, ctr_lim=calc_len, log_n_max=nd*2, ftype=ftype)
   ctr_tot += vs[i_int][i_ctr]
   t_el = time.time() - t_0
  nips = ctr_tot / t_el
  nips_arr[i_nd] = nips
 return {ftype: np.vstack((nd_arr, nips_arr))}

def records_path_all(data_all={}, fn_d='', ks=None, n_start=3, n_end=5e7, t_lim=1000, log_n_max=9000, ctr_lim=3e9, ftype=''): 
 '''
 calculates jump records table with jump_slope_calc for all (mult, add, divisors) values included in ks argument and write it to data_all dictionary under (m, a, d) keys
 if 'data_all' arg is None, create new dictionary
 if 'fn_d' arg is a valid file name, save it to fn_d binary file
 is ks is None, calculate for all entries in 'data_all' dictionary database.
 for diverging-type sequences (average slope for (m,a,d) > 0), write dummy values
 if 'recalc' in ftype: rcalculate already-present entries; otherwise, update
 '''

 print_type = 'print' if 'print' in ftype else ''
 return_dict = True if 'return' in ftype else False
 recalc_type = 'recalc' if 'recalc' in ftype else ('skip' if 'skip' in ftype else 'update')
 
 if fn_d != '' and fn_d in os.listdir():
  data_all = db_load(fn_d)
       
 if not (isinstance(ks, list) or isinstance(ks, tuple)):
  ks = db_keys(data_all)

 ctr_mad, t_0 = -1, time.time()
    
 t_0, ctr = time.time(), -1
 for mad in ks:

  t_lim_cur = t_lim[mad] if isinstance(t_lim, dict) and mad in t_lim.keys() else t_lim
  log_n_max_cur = log_n_max[mad] if isinstance(log_n_max, dict) and mad in log_n_max.keys() else log_n_max
  ctr_lim_cur = ctr_lim[mad] if isinstance(ctr_lim, dict) and mad in ctr_lim.keys() else ctr_lim

  ctr += 1
  
  if mad not in data_all.keys():
   data_all[mad] = {}
  
  data_present_rec = val_check(data_all[mad], keys_rec, mad=mad) 
  data_present_sl = val_check(data_all[mad], keys_inf, mad=mad) 

  if data_present_rec and recalc_type == 'skip':
   continue

  if data_present_sl:  
   slope_avg = data_all[mad]['slope_avg']
  else:
   vs_sl = avg_slope_calc(n_digits=222, calc_len=122, mad=mad, n_iter_lim=1e9, f_cond=None, t_lim=(0.5*t_lim_cur)**(3/5), exp_lim=2.0, sigma_thr=0.0, skip=51, ftype='')
   data_all[mad].update(vs_sl)
   slope_avg = vs_sl['slope_avg']
  
  if slope_avg > 0:
   data_all[mad].update({'path_records': []})
   continue
 
  print(f"{time.time() - t_0:.2f} sl:{slope_avg:.6f} mad:{mad}")

  data_prev = {x:data_all[mad][x] for x in keys_rec} if recalc_type == 'update' and data_present_rec else {}        
  vs = records_path_calc(mad, n_start=n_start, n_end=n_end, vals_add=data_prev, t_lim=t_lim_cur, log_n_max=log_n_max_cur, ctr_lim=ctr_lim_cur, ftype=print_type)
  rec_tbl, ns_min, ec_wrong = vs['path_records'], vs['cycle_ns_min'], vs['ec_wrong_path']
  jump_fit = jump_slope_calc(rec_tbl=rec_tbl, inds=(0.4,1.0), ftype='lin_fit') 
  rec_slope = jump_slope_calc(rec_tbl=rec_tbl, inds=(0.4,1.0), ftype='through_zero') 
  res_fit = jump_slope_calc(rec_tbl=rec_tbl, inds=(0.2,1.0), ftype='res_fit') 
  print(f"{time.time() - t_0:.2f} rec_sl: {rec_slope:.6f} fit: {jump_fit[0]:.6f} {jump_fit[1]:.6f} res_fit:{res_fit:.6f} {mad}")
  data_all[mad].update({'mad': mad, 'path_records': rec_tbl, 'ec_wrong_path': ec_wrong})
   
  if 'cycle_ns_min' not in data_all[mad].keys():
   data_all[mad].update({'cycle_ns_min': ns_min})
  else:
   cycle_ctrs_present = 'cycle_ctrs' in data_all[mad].keys()
   for n in ns_min:
    if n not in data_all[mad]['cycle_ns_min']:
     data_all[mad]['cycle_ns_min'].append(n)
     if cycle_ctrs_present:
      data_all[mad]['cycle_ctrs'].append(1)
      
   
  if sqrt(ctr) % 1 == 0 and fn_d != '' and 'save' in ftype:
   db_save(data_all, fn_d)
 
 if fn_d != '':
  db_save(data_all, fn_d)
	  
 return data_all if return_dict else None


#remaining: try to implement cycle ctrs here 
def records_path_calc(mad, n_start=3, n_end=1e20, vals_add={}, t_lim=100, log_n_max=10000, ctr_lim=1e9, ftype='print'):
 '''
 Calculate table of jump records, analogous to https://www.ericr.nl/wondrous/pathrecs.html
 Returns numbers, records of max(iter_log) and minimums of found cycles
 iterate over odd numbers from n_start
 if max(iter_log) > previous maximum, add number and max(iter log) to table
 '''
 n_end = int(n_end)
 mult, add, divisors = mad
 
 prev_data_present = val_check(vals_add, keys_rec, mad=mad)
 
 if prev_data_present:
  rec_tbl, ns_min = [copy.deepcopy(vals_add[x]) for x in ('path_records', 'cycle_ns_min')]
  n_start = rec_tbl[-1][0] + 1
 else:
  rec_tbl, ns_min = [], []  #cycle detection does not take much time, so no need to import ns_min from stats_range results
 
 ctr_calc = -1
 t_0 = time.time()
 n = n_start - 1
 ec_wrong = 0
 while True:
  n += 1
  ctr_calc += 1

  if sqrt(ctr_calc) % 1 == 0:
   t_cur = time.time() - t_0
   if t_cur > t_lim or (len(rec_tbl) > 0 and rec_tbl[-1][0] > n_end):
    break

  if any([n % d == 0 for d in divisors]):
   continue
  n_min_lim = 0 if 'full' in ftype else n-1
  vs_int = collatz_c_short(n, mult=mult, divisors=divisors, add=add, log_n_max=log_n_max, ctr_lim=ctr_lim, ns_min=ns_min, n_min_lim=n_min_lim)
  ctr, op_ctrs, n_start, n_last, jump, glide, end_cond = vs_int
  
  if end_cond in(2, 3):
   ec_wrong += 1
   print(f"number {n} not converged with end_cond {end_cond}")
   continue
  
  if end_cond in (0, 1, 5):
   if len(rec_tbl) == 0 or jump > rec_tbl[-1][1]:
    rec_tbl.append([n, jump])
    if 'print' in ftype:
     jump_str = str(jump) if log10(jump)<30 else f"{log10(jump):.3f}"
     print(f"{time.time() - t_0:.2f}\t{ctr_calc:.2e} {log10(jump)/log10(n):.6f} {n} {jump_str} ")

  if end_cond in (1, 4): #converged to a new cycle, from outside or from within #adjust hard-coded ctr_lim limitation according to operative memory
   vs_ccl = collatz_calc(n_last, mad=(mult, add, divisors), log_n_max=9000, ctr_lim=min(ctr_lim, 2e8), ftype='def') #in case of large cycles 
   ccl = vs_ccl[i_il]
   n_min = min(ccl) 
   if ccl[0] == ccl[-1] and n_min not in ns_min:
    ns_min.append(n_min)
    if 'cycle_ctrs' in vals_add.keys():
     cycle_ctrs.append(1)
    if 'print' in ftype:
     print(f"cycle added witn n_min {n_min}")
 
 vs_out = {'mad': mad, 'path_records': rec_tbl, 'cycle_ns_min': ns_min, 'ec_wrong_path': ec_wrong}
 
 return vs_out

def records_path_avg(mad=(3,1,(2,)), data_all={}, fn_d='', n_adds=5, ftype='print', mult=3, divisors=(2,)):
 
 '''
 Averages records with same mult and divisors but different addition values from a database
 
 Takes:
 mad: (mult, add, divisors) (add value is ignored), fn_d: filename of database, n_adds: number of additions values (<= number of entries with same mult and divisors)
 
 Merges tables into one:
 -Sorts starting numbers
 -If jump record with any addition is higher than all records with all possible additions for smaller numbers:
 --Include it in the table
 --Repeat with the new table
 
 Returns dictionary {k:v} where:
  'path_records_top': zipped list of numbers, path records and corresponding addition values, as defined above 
  'path_records_all': the same structure, but for all numbers and their records found for specified mult and divisors, sorted by starting number
  'nums_all', 'recs_all': recs_all,  'adds_all', 'ns_min_all': individual lists for each add value, grouped into lists
 
 '''

 if isinstance(mad, tuple) and len(mad) == 3:
  mult, _, divisors = mad

 if fn_d != '' and fn_d in os.listdir():
  data_all = db_load(fn_d)

 ks = db_keys(data_all, adds='all')
 ks_cur = [k for k in ks if k[0] == mult and k[2] == divisors]
 n_adds = len(ks_cur) if n_adds <= 0 else min(len(ks_cur), n_adds)
 
 nums_all, recs_all, adds_all, ns_min_all = [], [], [], []
 for i_add in range(n_adds):
  mad = ks_cur[i_add]  
  add = mad[1]
  path_recs_cur, ns_min = data_all[mad]['path_records'], data_all[mad]['cycle_ns_min']
  
  nums, recs = [x[0] for x in path_recs_cur], [x[1] for x in path_recs_cur]
  nums_all.append(nums)
  recs_all.append(recs)
  adds_all.append([add for n in nums])
  ns_min_all.append(ns_min)
 
 num_minmax = min([max(nums) for nums in nums_all])
 
 nums_fl = [item for sublist in nums_all for item in sublist] 
 recs_fl = [item for sublist in recs_all for item in sublist] 
 adds_fl = [item for sublist in adds_all for item in sublist] 
 
 i_srt = [x[0] for x in sorted(enumerate(nums_fl), key=lambda x:x[1])] 
 i_val = [i for i in i_srt if nums_fl[i] < num_minmax]
 nums_srt = [nums_fl[i] for i in i_val]
 recs_srt = [recs_fl[i] for i in i_val]
 adds_srt = [adds_fl[i] for i in i_val]
 
 nums_top, recs_top, adds_top = [nums_srt[0],], [recs_srt[0],], [adds_srt[0],]
 for i in range(len(nums_srt)):
  if nums_srt[i] > nums_top[-1]:
   if recs_srt[i] > recs_top[-1]:
    nums_top.append(nums_srt[i])
    recs_top.append(recs_srt[i])
    adds_top.append(adds_srt[i])
  else:
   if recs_srt[i] > recs_top[-1]:
    recs_top[-1] = recs_srt[i]
    adds_top[-1] = adds_srt[i]
  
 recs_max = list(zip(nums_top, recs_top, adds_top))
 recs_avg = list(zip(nums_srt, recs_srt, adds_srt))
 vs = {'path_records_top': recs_max, 'path_records_all': recs_avg, 
       'nums_all': nums_all, 'recs_all': recs_all, 'adds_all': adds_all, 'ns_min_all': ns_min_all}
 return vs #nums_top, recs_top, adds_top, nums_all, recs_all, adds_all, ns_min_all
 
# f_log = lambda x: log10(x) if x > 0 else (-inf if x == 0 else nan)
# f_arr = (lambda x: f_log(x)) if 'log' in val_types[i] else (lambda x: x)


#----------------------------------------------------------------------------------------------------------

def jump_slope_res(nums, recs, pwr=2.0, ftype='sum'):
 '''
 See jump_slope_calc for full description.
 Used in calculations of slope of log(jump record) vs log(starting number)
 Calculates fit metric for slope of log(jump) vs log(num).
 Fit metric is average of sum of (log10(jump) / log10(num**pwr)), minus 1.0

 '''
 jump_rec_res_raw = [10**(log10(recs[i]) - pwr*log10(nums[i])) for i in range(len(nums))] #[recs[i] / nums[i]**pwr for i in range(len(nums))]
 jump_rec_res = [float(min(max(x, 10**-300.0), 10**300.0)) for x in jump_rec_res_raw]
 if 'prod' in ftype: # <<1 for (3,1,(2), and 2.0), result is 1.941 which is obviously too low.
  metric = log10(prod(jump_rec_res)/len(jump_rec_res))
 else:
  metric = sum(jump_rec_res)/len(jump_rec_res) - 1.0
 return metric



def jump_slope_calc(rec_lst=None, nums=None, recs=None, mad=None, inds=(0,0), ftype='res_fit'):
 '''
 For any converging Collatz function, there seems to be a bound on relative maximum of iteration sequence ('jump') resulting from it:
 log(sequence maximum) / log(n_start) <= pwr, where for classic Collatz function, pwr seems to be very close to 2.0 (https://www.ericr.nl/wondrous/pathrecs.html)
 One way to calculate jump records is as follows:
 Iterate from 1 over all odd numbers (not divisible by any of divisor set)
 If sequence maximum is higher than all previously observed maximums:
  Append number and it's 'jump' to the record table
 This function calculates jump record table in this way, using short-type collatz function which does not calculate full iteration log and terminates calculations if sequence falls below previously calculated numbers, and slope fit to this table, according to ftype and inds.
 
 If nums and recs arguments are lists, starts from maximum of nums and appends new records to these lists
 
 Statistics:
  -Take part of record table bound by inds
  --direct indices if int
  --relative span if float: inds = int(table length * (inds)), e.g. 20 * (0.2, 0.8) = from 4 to 16
  --used to exclude first values which may have large deviations)
  -Calculate fit of log(n_max) / log(n_start):
  --linear fit ('lin_fit' in ftype)
  --linear fit through zero ('through_zero' in ftype)
  --residue fit of log_n_max - log_n_start**pwr, with jump_slope_res function ('res_fit' in ftype)
  --simple average otherwise
  
 '''
 
 if isinstance(rec_lst, list):
  nums, recs = [x[0] for x in rec_lst], [x[1] for x in rec_lst]
  
 len_all = len(nums)
 i_start, i_end = inds

 if i_end == 0 or not isfinite(i_end):
  i_end = len(nums)
 elif i_end < 0:
  i_end = len(nums) + i_end
 elif isinstance(i_end, float):
  i_end = int(i_end*len_all)
 if isinstance(i_start, float):
  i_start = int(i_start*len_all)

 nums_slc, recs_slc = nums[i_start:i_end], recs[i_start:i_end]  
 log_nums_slc =  [log10(nums_slc[i]) for i in range(len(nums_slc))]
 log_recs_slc = [log10(recs_slc[i]) for i in range(len(nums_slc))]
 jumps_slc = [log_recs_slc[i] / log_nums_slc[i] for i in range(len(nums_slc))]
 slope_rough = max(jumps_slc)#sum(jumps_slc) / len(jumps_slc)

 if 'res_fit' in ftype:
  j_sl_fit = lambda x: log10(jump_slope_res(nums_slc, recs_slc, x, ftype='prod' if 'prod' in ftype else '') + 1)
  jump_rec_slope = scp.optimize.newton(j_sl_fit, x0=slope_rough) 
  return jump_rec_slope
 elif 'lin_fit' in ftype:
  return np.polyfit(log_nums_slc, log_recs_slc, deg=1)
 elif 'through_zero' in ftype:
  lin_fit = lambda x, y: x*y
  return scp.optimize.curve_fit(lin_fit, np.array(log_nums_slc), np.array(log_recs_slc))[0][0]
 else:
  return slope_rough
 
 


def drop_block_stats_all(data_all, len_block=111, t_lim=(1000,), fn_d='', ks=None, ks_pass=None, 
                         nd=100, skip=51, n_bins=100, n_tails=20, nd_round=8, ftype=''):
 
 '''
 Wrapper for drop_block_stats function to calculate drop block stats for entries in a database data_all with keys equal to (mult, add, divisors) for different Collatz functions
 if 'data_all' arg is None, create new dictionary
 if 'fn_d' arg is a valid file name, save it to fn_d binary file
 is ks is None, calculate for all entries in 'data_all' dictionary database.
 if 'recalc' in ftype: rcalculate already-present entries; otherwise, update
 '''
 
 print_type = 'print' if 'print' in ftype else ''
 return_dict = True if (not isinstance(data_all, dict) or 'return' in ftype) else False

 if fn_d != '' and fn_d in os.listdir():
  with open(fn_d, 'rb') as d:
   data_all = pickle.load(d)
   
 if not isinstance(data_all, dict):
  data_all = {k:{} for k in ks}
    
 if not (isinstance(ks, list) or isinstance(ks, tuple)):
  ks = db_keys(data_all)


 ctr_mad, t_0 = -1, time.time()
 
 if isinstance(t_lim, tuple) or isinstance(t_lim, list) and len(t_lim) > 1:
  t_lim_neg, t_lim_pos = t_lim[:2]
 elif isinstance(t_lim, float) or isinstance(t_lim, int):
  t_lim_neg = t_lim
  t_lim_pos = t_lim_neg/100

 for mad in ks:
  ctr_mad += 1
  if mad not in data_all.keys():
   data_all[mad] = {}
  if 'slope_avg' not in data_all[mad].keys():
   vs_sl = avg_slope_calc(n_digits=222, calc_len=222, mad=mad, n_iter_lim=1e9, f_cond=None, t_lim=(0.5*t_lim)**(3/5), exp_lim=2.0, sigma_thr=0.0, skip=51, ftype='')
   d['slope_avg'] = vs_sl['slope_avg']
  if isinstance(t_lim, dict):
   t_lim_cur = t_lim[mad]
  else:
   t_lim_cur = t_lim_pos if data_all[mad]['slope_avg'] > 0 else t_lim_neg
  len_block_cur = len_block[mad] if isinstance(len_block, dict) else len_block 
  sk = random.randint(int(skip*0.7), int(skip*1.3)) 

  if t_lim_cur <= 0:
   continue
  if 'lim_stats' in data_all[mad].keys() and not ('recalc' in ftype or 'upd' in ftype):
   continue  
  if (isinstance(ks_pass, list) or isinstance(ks_pass, tuple)) and mad in ks_pass:
   continue
  
  vs_sl = avg_slope_calc(n_digits=nd, calc_len=len_block_cur, mad=mad, n_iter_lim=3e9, f_cond=None, t_lim=t_lim_cur, skip=sk, ftype='drop')
  drops = vs_sl['drops']
  dr_vals, dr_cts = hist_with_means(drops, n_bins=n_bins, n_tails=n_tails, nd_round=nd_round, ftype='')
 
  if 'lim_stats' in data_all[mad].keys() and data_all[mad]['lim_stats']['len_block'] == len_block_cur and 'upd' in ftype:
   dr_vals_prev, dr_cts_prev = data_all[mad]['lim_stats']['dr_vals'], data_all[mad]['lim_stats']['dr_cts']
   dr_vals_all, dr_cts_all = merge_hists_brute(dr_vals, dr_cts, dr_vals_prev, dr_cts_prev, n_bins=-1, n_tails=-1, nd_round=-1, ftype='')   #use with care !
   lim_stats = {'len_block': len_block_cur, 'dr_vals': dr_vals_all, 'dr_cts': dr_cts_all}
   str_status = 'updated'
  else:
   lim_stats = {'len_block': len_block_cur, 'dr_vals': dr_vals, 'dr_cts': dr_cts}
   str_status = 'recalculated' if 'lim_stats' in data_all[mad].keys() else 'calculated anew'
  data_all[mad]['lim_stats'] = lim_stats
   
  print(f"{ctr_mad} {time.time() - t_0:.2f} {drops.size} {mad} status:{str_status}")
 
  if sqrt(ctr_mad) % 1 == 0 and fn_d != '' and 'save' in ftype:
   with open(fn_d, 'wb') as d:
    pickle.dump(data_all, d) 

 if fn_d != '' and 'save' in ftype:
  with open(fn_d, 'wb') as d:
   pickle.dump(data_all, d) 

 return data_all if return_dict else None


def log_arr_calc(mad=(3, 1, (2,)),n_digits=2000, n_sample=20, log_slice=100, ctr_lim=1e7, t_lim=1000, ftype='c_full'):
 '''
 Used for visualizations and for analysis of very long iteration logs 
 Calculates array of iteration logs
 Creates random sample of n_sample numbers n_digits long
 Calculates Collatz iteration log up to ctr_lim iterations
 Returns log(x) for x in resulting arrays
 Returns continuous log or each log_slice'th iteration in each log
 For large log_slice numbers, uses calculations, writes only end number of each sub-log

 '''
 c_type = 'py_short' if 'py_short' in ftype else 'c_full'
 
 ctr_lim = int(ctr_lim)
 log_n_max = int(n_digits*2)
 if c_type == 'c_full':
  log_buf = iter_log_init(log_n_max, ctr_lim) 
 else:
  log_buf = None
 
 t_0 = time.time()
 logs = []
 lens_min = []
 for i_c in range(n_sample):
  n_0 = range_gen((n_digits, n_digits+1), 1, [])[0] 
  vs_i, il, ol, _ = collatz(n_0, mad=mad, log_n_max=n_digits*2, ctr_lim=ctr_lim, log_slice=log_slice, num_buf=log_buf, ftype=c_type) 
  log_log = np.array([log10(x) for x in il])
  len_min_cur = np.where(log_log > n_digits/5)[0][-1]
  if i_c > 1 and len_min_cur * log_slice < ctr_lim:
   ctr_lim = len_min_cur * log_slice
  lens_min.append(len_min_cur)
  logs.append(log_log)
  if 'print' in ftype and sqrt(i_c) % 1 == 0:
   print(f"{i_c} {time.time() - t_0:.2f}")
  if time.time() - t_0 > t_lim:
   break

 if c_type == 'c_full':
  buf_size = lib.free_iter_log(byref(log_buf)) #1.5G per 3000, 1e7

 min_len = min(lens_min)
 log_log_arr = np.vstack([x[:min_len] for x in logs])
 return log_log_arr


def log_dispersion_all(data_all=None, fn_d='', ks=None, n_digits=1000, n_sample=3000, ctr_lim=1e3, t_lim=10, ftype=''):
 
 return_dict = True if (not isinstance(data_all, dict) or 'return' in ftype) else False
 print_type = 'print' if 'print' in ftype else ''
 
 if fn_d != '' and fn_d in os.listdir():
  with open(fn_d, 'rb') as d:
   data_all = pickle.load(d)
   
 if not isinstance(data_all, dict):
  data_all = {k:{} for k in ks}
    
 if not (isinstance(ks, list) or isinstance(ks, tuple)):
  ks = db_keys(data_all)

 t_0 = time.time()
 ctr_mad = 0
 
 for mad in ks:
  ctr_mad += 1
  
  n_digits_cur = n_digits[mad] if isinstance(n_digits, dict) and mad in n_digits.keys() else n_digits
  n_sample_cur = n_sample[mad] if isinstance(n_sample, dict) and mad in n_sample.keys() else n_sample
  ctr_lim_cur = ctr_lim[mad] if isinstance(ctr_lim, dict) and mad in ctr_lim.keys() else ctr_lim
  t_lim_cur = t_lim[mad] if isinstance(t_lim, dict) and mad in t_lim.keys() else t_lim
   
  if 't_calc_disp' in data_all[mad].keys() and data_all[mad]['t_calc_disp'] > t_lim_cur * 0.7 and 'recalc' not in ftype:
   continue
   
  slope_avg = data_all[mad]['slope_avg'] if 'slope_avg' in data_all[mad].keys() else nan
  
  vs = log_dispersion_calc(mad=mad, n_digits=n_digits_cur, n_sample=n_sample_cur, ctr_lim=ctr_lim_cur, t_lim=t_lim_cur, slope_avg=slope_avg, ftype=print_type)
 
  data_all[mad].update({'log_dispersion': vs['log_dispersion'], 't_calc_disp': vs['t_calc_disp']})
  if not isfinite(slope_avg) and isfinite(vs['slope_avg']):
   data_all[mad]['slope_avg'] = vs['slope_avg']
  
  print(f"{time.time() - t_0:.3f} {ctr_mad} {vs['log_dispersion']:<10.6f} {vs['slope_avg']:.6f} {mad}")
  
  if sqrt(ctr_mad) % 1 == 0 and fn_d != '' and 'save' in ftype:
   with open(fn_d, 'wb') as d:
    pickle.dump(data_all, d) 

 if fn_d != '' and 'save' in ftype:
  with open(fn_d, 'wb') as d:
   pickle.dump(data_all, d) 
 
 return data_all if return_dict else None



def log_dispersion_calc(mad=(3,1,(2,)), n_digits=1000, n_sample=3000, ctr_lim=1e3, t_lim=10, slope_avg=nan, ftype='print'): 
 print_type = ' print' if 'print' in ftype else '' 
 coll_type = 'py_short' if 'c_full' not in ftype else 'c_full'

 t_0 = time.time()
 if ctr_lim < 0: #time-limited calc
  slope_avg = avg_slope_calc(n_digits=60, calc_len=122, mad=mad, n_iter_lim=1e9, f_cond=None, t_lim=max(1,0.5*t_lim**(3/5)), exp_lim=2.0, sigma_thr=0.0, skip=51, ftype='')['slope_avg'] if not isfinite(slope_avg) else slope_avg
  nd = n_digits if n_digits > 0 else 500
  ctr_lim_max = int(abs(0.5* nd / slope_avg ))
  nips = collatz_speed_calc(mad=mad, n_digits=nd, ctr_lim=10000, n_rep=5, log_slice=10, ftype=coll_type)
  ctr_lim = max(1e3, min(ctr_lim_max, (t_lim*nips)/n_sample))
  n_sample=9000
 
 if n_digits < 0:
  if isfinite(slope_avg):
   n_digits = int(abs(slope_avg*ctr_lim) + 2*ctr_lim**0.5 + 100)
  else:
   n_digits=1000 
 
 log_slice = max(10, int(ctr_lim/1000))
 log_log_arr = log_arr_calc(mad=mad, n_digits=n_digits, n_sample=n_sample, log_slice=log_slice, ctr_lim=ctr_lim, t_lim=t_lim, ftype=coll_type + ' ' + print_type)
 ctrs = np.arange(log_log_arr.shape[1])*log_slice
 stds = log_log_arr.std(axis=0)

 std_sqr_fit = np.polyfit(np.sqrt(ctrs), stds, 1)
 slope, covar = scp.optimize.curve_fit(lambda x, m: m*x, np.sqrt(ctrs), stds) 
 
 if 'viz' in ftype:
  std_fit = (np.sqrt(ctrs)*std_sqr_fit[0] + std_sqr_fit[1])
  sqr_fit = np.sqrt(ctrs) * float(slope)
  plt.ion()
  plt.show()
  plt.clf()
  plt.plot(np.sqrt(ctrs), stds)
  plt.plot(np.sqrt(ctrs), std_fit)
  plt.plot(np.sqrt(ctrs), sqr_fit)
  plt.grid(True)
  print(f"disp = sqrt(ctr)*{float(slope):.5f}; linear fit: disp = sqrt(ctr) * {std_sqr_fit[0]:.5f} + {std_sqr_fit[1]:.5f}")
 
 log_disp = float(std_sqr_fit[0]) if 'fit' in ftype else float(slope)
 vs = {'log_dispersion': log_disp, 't_calc_disp': time.time() - t_0, 'slope_avg': slope_avg}
 if 'full' in ftype:
  vs['log_disp_arr'] = np.array((ctrs, stds))
 return vs

def stats_range_all():
 return None 

#remaining: add log_range check in previous data detect
def stats_range(log_range=(12, 12.5), mad=(), log_n_max=1000, 
                ctr_lim=1e9, n_sample=100500, t_lim=inf, nums=[], table_len=1000, rec_min_rel=100.0,
                ftype='odd', fn='', f_cond=None, vals_add={}, ns_min_add=[], mult=3, add=1, divisors=(2,)):
 '''
 Calculate statistics for collatz sequences defined by 'mult', 'add', and 'divisors', for numbers with logarithms in log_range.
 Terminate each calculation if iteration count exceeds ctr_lim, or number exceeds 10**log_n_max, or other end conditions are reached (see collats docstring)
 Perform calculations for 'n_sample' random numbers, or 't_lim' seconds, or numbers from nums list if not empty (externally-generated sample), or straight sequence starting from 10**log_range[0]
 
 Returns:
 -integer-type stats for all numbers in the sample, or numbers passing under f_cond argument (e.g. slope > -0.01), or dummy value
 -float-type stats (the same)
 -tuple of records of slope, completeness and log jump (if record writing is specified else empty tuple)
 -list of all found cycles, sorted by smallest member
 -list of biggest members in cycles
 -list of smallest members in cycles
 -numpy array of fractions of numbers which converged to each cycle, grew beyond log_n_max ('diverged') or iterated until ctr_lim ('undefined') (last 2 positions)
 
 ftype codes: 
 'odd' - use only odd starting numbers; 
 'high' or 'cycles': do not save and return stats for each number (used for cycle searches with large sample sizes)
 'records' - write top 'table len' records for slope, jump, completeness, if number > max(rec_min, 10**log_range(0))
 'classic' - use only for long 3n+a/2 high cycle searches (breaks if high cycle detected, saves the number)
 
 -if f_cond function is specified, includes only logs which pass under f_cond (in example, completeness or slope in specified range: f_cond = lambda x: x[i_float][i_slope] > 0.01 on x as collatz_calc function output)
 cycle_vals: cycles, cycle ctrs, divergence counter, undefined counter, imported from previous calculations if needed
 '''

 #----------------------------------preparations---------------------------

 if isinstance(mad, tuple) and len(mad) == 3:
  mult, add, divisors = mad
 mad = (mult, add, divisors)
 
 n_sample = int(n_sample) 
 classic = (mult == 3 and add == 1 and divisors[0] == 2) or 'classic' in ftype
 high = 'high' in ftype or 'cycles' in ftype
 straight = 'straight' in ftype
 print_type = 'print' in ftype
 odd_type = 'odd' in ftype
 full_type = 'full' in ftype and not high
 ds_range_gen = divisors if odd_type else []
 found_high = False 
 ftype_records = 'record' if 'records_asc' in ftype else ''
 
 #----------------------------number sample preparation--------------------
 
 if len(nums) > 0:
  ns = nums
  len_ns = len(ns)
 else:
  n_start, n_end = int(10**log_range[0]), int(10**log_range[1])
  if (n_end - n_start) * (div_prob_calc(divisors)[0] if odd_type else 1.0) < n_sample:
   straight = True 
  if not (straight or high): 
   ns = range_gen(log_range=log_range, n_sample=n_sample, divisors=ds_range_gen)
   len_ns = len(ns)
  else:
   len_ns = inf
 
 #----------------import values from previous calculations (e.g. with different log_range)

 prev_data_present = val_check(vals_add, keys_range, mad=mad)
 
 if prev_data_present: #update previously-calculated values
  dp = [copy.deepcopy(vals_add[x]) for x in ('cycle_ns_min', 'cycle_ctrs', 'div_ctr', 'undef_ctr', 'stat_list', 'records', 't_calc_range')]
  ns_min, cycle_ctrs, div_ctr, undef_ctr, stat_list, recs, t_calc_range = dp
  ns_min_append = [n for n in ns_min_add if n not in ns_min]
  ns_min += ns_min_append
 else:
  ns_min = [n for n in ns_min_add]
  cycle_ctrs = [0 for n in ns_min]
  div_ctr, undef_ctr, stat_list, recs, t_calc_range = 0, 0, [], Records(n_max=table_len), 0.0 #record table, defined in collatz_aux

 if 0 not in ns_min: #ns_min could be imported from elsewhere (records_path_avg)
  ns_min = [0,] + ns_min
  cycle_ctrs = [0,] + cycle_ctrs  

 #----------------------------------------------------------------------
 
 i_n, cond = -1, True
 t_0 = time.time()
 ctr_calc = 0 #sum(cycle_ctrs) + div_ctr + undef_ctr
 
 #--------------------------main cycle----------------------------------
 
 while cond:
  i_n += 1
  if high:
   n = range_gen(log_range, 1, ds_range_gen)[0]
  elif straight:
   n = n_start + i_n
  else: 
   n = ns[i_n]

  if straight and odd_type and any([n % d == 0 for d in divisors]):
   continue
  
  vs = collatz_calc(n, mad=(mult,add,divisors), ns_min=ns_min, log_n_max=log_n_max, ctr_lim=ctr_lim, ftype='short')
  end_cond, n_start, n_last = vs[i_int][i_end_cond], vs[i_int][i_n_start], vs[i_int][i_n_last]
         
  if end_cond == -1: #some error in calculations
   continue

  elif end_cond == 0: #converged to a known cycle
   i_cycle = ns_min.index(n_last)
   cycle_ctrs[i_cycle] += 1

  elif end_cond in(1, 4): #converged to a new cycle, from outside or from within #adjust hard-coded ctr_lim limitation according to operative memory
   vs_ccl = collatz_calc(n_last, mad=(mult, add, divisors), log_n_max=log_n_max, ctr_lim=min(ctr_lim, 2e8), ftype='def') #in case of large cycles 
   n_min = min(vs_ccl[i_il])
   if n_min not in ns_min:
    ns_min.append(n_min)
    cycle_ctrs.append(1)     
   else:
    i_cycle = ns_min.index(n_min)
    cycle_ctrs[i_cycle] += 1  
   vs = collatz_calc(n, mad=(mult,add,divisors), ns_min=ns_min, log_n_max=log_n_max, ctr_lim=ctr_lim, ftype='short') #if was within a cycle, recalc until n_min
   end_cond, n_start, n_last = vs[i_int][i_end_cond], vs[i_int][i_n_start], vs[i_int][i_n_last]
   if classic and n_min > 10000:
    print ('non-trivial cycle found!!!11')
    num_save(n_start, mad)

  elif end_cond == 2: #diverged
   div_ctr += 1
  elif end_cond == 3: #undefined (hit ctr_lim without diverging or converging)
   undef_ctr += 1
    
  if full_type and (f_cond == None or f_cond(vs)): #if a condition is specified, append only stats which pass under it
   stat_list.append(vs[:2]) #append integer and float-type stats
   
  #only numbers which converged to a cycle from outside, and are more than rec_min_rel * end number
  if end_cond in (0, 1) and n_start > rec_min_rel*n_last and table_len > 0:
   recs.upd(vs, ftype=ftype_records)   
   
  ctr_calc += 1
  t_cur = time.time() - t_0

  if print_type and sqrt(ctr_calc) % 50 == 0:  
   if len(recs.slopes) != 0 and len(recs.compls) != 0 and len(recs.jumps) != 0 and table_len > 0:
    print(f"{t_cur:.2f} {ctr_calc} records slope, compl, jump: {recs.slopes[-1][i_float][i_slope]:.6f} {recs.compls[-1][i_float][i_compl]:.6f} {recs.jumps[-1][i_float][i_log_jump]:.6f}")
   else:
    print(f"{t_cur:.2f} {ctr_calc}")
  
  if t_cur > t_lim:
   break
  elif ((straight or high) and ctr_calc >= n_sample):
   break
  elif i_n >= len_ns-1:
   break
   
 #-------------------------------prepare other return values------------------
   
 t_fin = time.time() - t_0 
 
 vs_out = {'mad': mad}
 vs_out.update({'cycle_ns_min': ns_min, 'cycle_ctrs': cycle_ctrs, 'div_ctr': div_ctr, 'undef_ctr': undef_ctr})
 vs_out.update({'stat_list': stat_list, 'records': recs, 't_calc_range': t_fin + t_calc_range})  
 
 return vs_out


def num_save(n_start, mad):
 fn_found = mad_to_str(mad) + '_nums_found.bin'
 if fn_found not in os.listdir():
  nums_found = []
 else:
  with open('nums_found.bin', 'rb') as d:
   nums_found = pickle.load(d)
     
 nums_found.append(n_last)
 with open('nums_found.bin', 'wb') as d: #tested OK on 10**(10**7), size = 4MB
  pickle.dump(nums_found, d)
 return None
 
 
def stats_to_arr(stat_list, inds, val_types=[]):
 vs = []
 types_specified = len(val_types) == len(inds)
 f_log = lambda x: log10(x) if x > 0 else (-inf if x == 0 else nan)
 for i in range(len(inds)):
  inds_cur = inds[i]
  if types_specified:
   f_arr = (lambda x: f_log(x)) if 'log' in val_types[i] else (lambda x: x)
  else:
   f_arr = (lambda x: f_log(x)) if inds_cur[0] == 0 and inds_cur[1] in (i_n_start, i_n_last, i_jump) else (lambda x: x)
  vs_cur = [f_arr(x[inds_cur[0]][inds_cur[1]]) for x in stat_list]
  vs.append(vs_cur)
 stats_arr = np.vstack(vs)
 return stats_arr
 

def log_n_max_calc(slope, disp, sigma=4.0):
 '''
 Calculates maximum expected deviation of number 'in the wrong direction',
 Upwards if converging, downwards if diverging
 In dispersion-dominated mode, these values do not depend on log(n_start)
 Can be used to calculate log_n_max in stats_all to save calc time, and in crude max jump calculation (assuming gaussian log dispersion which may be wrong at the tails)
 '''
 log_n_max = 0.75*sigma**2*disp**2/slope
 ctr_at_max_dev = (0.5*sigma*disp /slope)**2
 return abs(log_n_max), ctr_at_max_dev
  

def jump_lim_crude(log_n, slope, disp, ftype=''):
 if log_n >= 30 or 'approx' in ftype:
  f_opt = lambda x: cdf_exp(x) + log_n
 else:
  f_opt = lambda x: log10(scp.stats.norm.cdf(x)) + log_n
 sigma = scp.optimize.newton(f_opt, x0=-1.0)
 log_n_max, ctr_exp_lim = log_n_max_calc(slope, disp, sigma)
 jump_lim = (log_n + log_n_max) / log_n
 return jump_lim


#add update/recalc ftype recognition
#here, recalc = recalc all stat types, update = update all possible, skip = skip if all values are present
def stats_calc_all(data_all=None, fn_d='', ks=None, sigma_disp=6.0, n_sample_cycles=2e4, log_n_max=9000, log_range_max=18,
                   n_records=100, rec_min_rel=1e2, pathrec_max=1e7, t_lim_slope=5, t_lim_records=2, t_lim_range=10, slope_sigma_thr=1e-8, 
                   slope_calc_len=122, slope_skip_lim=51, ftype='print'):

 print_type = 'print_full' if 'print_full' in ftype else ('print' if 'print' in ftype else '')
 return_dict = True if 'return' in ftype else False
 recalc_type = 'recalc' if 'recalc' in ftype else ('skip' if 'skip' in ftype else 'update')
 
 if fn_d != '' and fn_d in os.listdir():
  data_all = db_load(fn_d)
       
 if not (isinstance(ks, list) or isinstance(ks, tuple)):
  ks = db_keys(data_all)

 t_0 = time.time()
 ctr_mad = ctr_calc = 0
 
 for mad in ks:
  ctr_mad += 1
  
  slope_calc_len_cur = slope_calc_len[mad] if isinstance(slope_calc_len, dict) and mad in slope_calc_len.keys() else slope_calc_len
  sigma_disp_cur = sigma_disp[mad] if isinstance(sigma_disp, dict) and mad in sigma_disp.keys() else sigma_disp
  n_sample_cycles_cur = n_sample_cycles[mad] if isinstance(n_sample_cycles, dict) and mad in n_sample_cycles.keys() else n_sample_cycles
  log_n_max_cur = log_n_max[mad] if isinstance(log_n_max, dict) and mad in log_n_max.keys() else log_n_max
  log_range_max_cur = log_range_max[mad] if isinstance(log_range_max, dict) and mad in log_range_max.keys() else log_range_max
  n_records_cur = n_records[mad] if isinstance(n_records, dict) and mad in n_records.keys() else n_records
  rec_min_rel_cur = rec_min_rel[mad] if isinstance(rec_min_rel, dict) and mad in rec_min_rel.keys() else rec_min_rel
  pathrec_max_cur = pathrec_max[mad] if isinstance(pathrec_max, dict) and mad in pathrec_max.keys() else pathrec_max
  t_lim_slope_cur = t_lim_slope[mad] if isinstance(t_lim_slope, dict) and mad in t_lim_slope.keys() else t_lim_slope
  t_lim_records_cur = t_lim_records[mad] if isinstance(t_lim_records, dict) and mad in t_lim_records.keys() else t_lim_records
  t_lim_range_cur = t_lim_range[mad] if isinstance(t_lim_range, dict) and mad in t_lim_range.keys() else t_lim_range
  slope_sigma_thr_cur = slope_sigma_thr[mad] if isinstance(slope_sigma_thr, dict) and mad in slope_sigma_thr.keys() else slope_sigma_thr
  slope_skip_lim_cur = slope_skip_lim[mad] if isinstance(slope_skip_lim, dict) and mad in slope_skip_lim.keys() else slope_skip_lim
  
  
  if mad not in data_all.keys():
   data_all[mad] = {}
  
  data_present_rec = val_check(data_all[mad], keys_rec, mad=mad) 
  data_present_sl = val_check(data_all[mad], keys_inf, mad=mad) 
  data_present_rng = val_check(data_all[mad], keys_range, mad=mad) 
  
  all_data_present = data_present_rec and data_present_sl and data_present_rng
  any_data_present = data_present_rec or data_present_sl or data_present_rng

  if all_data_present and recalc_type == 'skip':
   print(f"{ctr_mad} {ctr_calc} {time.time() - t_0:.3f} s" + stats_to_str(data_all[mad]))
   continue
  
  ctr_calc += 1
  vals_prev = data_all[mad] if recalc_type == 'update' and any_data_present else {}

  vs = stats_calc(mad=mad, vals_add=vals_prev, ftype=recalc_type + ' ' + print_type,
                  sigma_disp=sigma_disp_cur, n_sample_cycles=n_sample_cycles_cur, log_n_max=log_n_max_cur, 
                  log_range_max=log_range_max_cur, n_records=n_records_cur, rec_min_rel=rec_min_rel_cur, pathrec_max = pathrec_max_cur,
                  t_lim_slope=t_lim_slope_cur, t_lim_records=t_lim_records_cur, t_lim_range=t_lim_range_cur,
                  slope_sigma_thr=slope_sigma_thr_cur, slope_calc_len=slope_calc_len_cur, slope_skip_lim=slope_skip_lim_cur)
                
  data_all[mad].update(vs)
  print(f"{ctr_mad} {ctr_calc} {time.time() - t_0:.3f} s" + stats_to_str(data_all[mad]))
  
  if sqrt(ctr_calc) % 1 == 0 and fn_d != '' and 'save' in ftype:
   db_save(data_all, fn_d)
  
 if fn_d != '' and 'save' in ftype:
  db_save(data_all, fn_d) 

 return data_all if return_dict else None


def stats_calc(mad=(3,1,(2,)), sigma_disp=6.0, n_sample_cycles=2e4, log_n_max=9000, log_range_max=18, n_records=100, rec_min_rel=1e2, vals_add={}, pathrec_max=1e7,
               t_lim_slope=5, t_lim_records=2, t_lim_range=10, slope_sigma_thr=1e-8, slope_calc_len=122, slope_skip_lim=51, ftype='print'):
               
 '''
 Calculate all stats for Collatz sequences defined by parameters mult, divisors, add
 
 -Calculates average slope with slope_inf_calc
 -Searches for cycles and calculates fractions of numbers which 
 --converge to cycles, 
 --apparently diverge to log_n_max ([0] if divergent-type (slope_avg > 0), [1] otherwise)
 --remain undefined after ctr_lim iterations
 -Calculates records of slope, completeness and jump for numbers used in cycle search
 -Calculates fraction of apparently-divergent and undefined numbers for numbers with n_digits included in log_range_div (uses log_n_max[1])
 
 -Saves stats in dictionary format:
 dict_keys(['mult', 'divisors', 'add', 'slope_avg', 'slope_sigma', 'compl_avg', 'compl_sigma', 
 'slope_max', 'compl_max', 'jump_max', 'records_slope', 'records_compl', 'records_jump', 
 'cycle_ns_min', 'cycle_ns_max', 'cycle_lens', 'cycle_compls', 'cycles', 'cycle_freqs', 'ctrs', 'divergence_stats'])
 #ctrs are raw counters outputted by stats_range: [converged to a cycle] + [divergent, undefined]
 '''
 
 recalc_type = 'recalc' if 'recalc' in ftype else ('skip' if 'skip' in ftype else 'update')
 
 if recalc_type == 'recalc':
  data_present_slope = data_present_pathrec = data_present_range = False 
 else:
  data_present_slope = val_check(vals_add, keys_inf, mad=mad) and (vals_add['len_block'] == slope_calc_len and (vals_add['t_calc_slope'] > t_lim_slope*0.7 or vals_add['slope_std'] < slope_sigma_thr*1.3) )
  data_present_pathrec = val_check(vals_add, keys_rec, mad=mad)
  data_present_range = val_check(vals_add, keys_range, mad=mad)

 results = {}
 t_0 = time.time()
 print_type = 'print' if 'print_full' in ftype else ''
 n_sample_cycles = int(n_sample_cycles)
  
 #------asymptotic statistics calculation--------
 
 if data_present_slope:
  stats_inf = {x:vals_add[x] for x in keys_inf}
 else: 
  stats_prelim = avg_slope_calc(n_digits=500, calc_len=301, mad=mad, n_iter_lim=1e9, t_lim=1.0, exp_lim=3.0)
  slope_prelim, disp_prelim = stats_prelim['slope_avg'], stats_prelim['log_dispersion'] #usually accurate to !2e-4 and 5e-3
  #calculate OOMAG range adaptively based on slope and dispersion
  nd_slope = int( 1.3*( abs(slope_calc_len * slope_prelim ) + abs(sqrt(slope_calc_len) * disp_prelim)) ) + 50 
  stats_inf = avg_slope_calc(mad=mad, n_digits=nd_slope, calc_len=slope_calc_len, skip=slope_skip_lim, n_iter_lim=1e9, t_lim=t_lim_slope, 
                             exp_lim=3.0, sigma_thr=slope_sigma_thr, ftype=print_type)
  
  results.update(stats_inf)  
 
 slope_avg, log_disp = stats_inf['slope_avg'], stats_inf['log_dispersion']
  
 if 'print' in ftype:
  print(f"{time.time() - t_0:.2f} asymptotic values:")
  op_freq_str = ''
  op_freq_std_str = ''
  for i in range(len(stats_inf['op_freqs'])):
   op_freq_str += f"{stats_inf['op_freqs'][i]:.6f} "
   op_freq_std_str += f"{stats_inf['op_freq_stds'][i]:.6f} "
  print(f"{time.time() - t_0:.1f} avg slope (sigma), compl, disp \t{stats_inf['slope_avg']:<10.6f} ({stats_inf['slope_std']:.6f}) {stats_inf['compl_avg']:.6f} {stats_inf['log_dispersion']:<7.4f}")
  print(f"op_freqs (sigmas) \t {op_freq_str} ({op_freq_std_str})")

 #-----------path records calculation---------------

 if slope_avg < 0:
  dp_rec = {x: vals_add[x] for x in keys_rec} if data_present_pathrec else {}
  vs_rec = records_path_calc(mad, n_start=3, n_end=pathrec_max, vals_add=dp_rec, t_lim=t_lim_records, log_n_max=log_n_max, ctr_lim=3e9, ftype=print_type)
  ns_min = vs_rec['cycle_ns_min']
 else:
  vs_rec = {'path_records':[], 'cycle_ns_min':[]}
  ns_min = []
  
 results.update(vs_rec) #take care not to overwrite ns_min if this is moved after stats_range
 
 
 #----------calculating adaptive limits for cycle detection/stats calculation
 
 log_n_max_disp, ctr_at_max_disp = log_n_max_calc(slope_avg, log_disp, sigma=sigma_disp)
 log_n_max_adj = min(log_n_max, int(ceil(log_n_max_disp) + log_range_max) + 20) if slope_avg > 0 else log_n_max
 #log_range_max: if GCF is diverging, set max number so that divergence probability is < 1.5-2sigma 
 log_range_max_adj = min(log_n_max_disp * (1.5 / sigma_disp)**2 + 3, log_range_max) if slope_avg > 0 else log_range_max
 if 'print' in ftype:
  print(f"{time.time() - t_0:.2f} adjusted limits: log_n_max {log_n_max_adj}; log_range_max {log_range_max_adj:.1f}")
 div_mult = ceil(1.3/div_prob_calc(mad[2])[0]) + 1
 
 #---------caclulating cycle statistics and record values-----------------

 nums_1 = [i for i in range(1, n_sample_cycles*div_mult) if not any([i % d == 0 for d in mad[2]])][:int(n_sample_cycles/2)]
 nums_1 = [n for n in nums_1 if n < 10**(log_range_max_adj + 0.3)]
 log_max_num = log10(nums_1[-1])
 nums_2 = range_gen((log10(max(nums_1)), log_range_max_adj), divisors=[], n_sample=int(n_sample_cycles/2)) if log_range_max_adj > log_max_num else []
 nums = nums_1 + nums_2
 #remaining: add range and 'already calculated nums' check here (not a small task, better to calculate cycle frequencies vs oomag elsewhere) 
 #in this case, adds only stats with sparsely sampled range
 dp_ccl = {x:vals_add[x] for x in keys_range} if data_present_range else {} 
 stats_r = stats_range(mad=mad, log_n_max=log_n_max_adj, ctr_lim=1e9, t_lim=t_lim_range, 
                       nums=nums, table_len=n_records, rec_min_rel=rec_min_rel, ftype=print_type,
                       vals_add=dp_ccl, ns_min_add = ns_min)
 results.update(stats_r)

 if 'print' in ftype:
  cycle_info = cycle_info_calc(stats_r)
  div_ctr, undef_ctr = [stats_r[k] for k in ('div_ctr', 'undef_ctr')]
  cycle_ctrs, ns_min, compls, frs, lens, ns_max = [cycle_info[k] for k in ('cycle_ctrs', 'cycle_ns_min', 'cycle_compls', 'cycle_freqs', 'cycle_lens', 'cycle_ns_max')]
  str_ccl = f"{time.time() - t_0:.3f} cycles: calculated total {sum(cycle_ctrs) + div_ctr + undef_ctr} (converged {sum(cycle_ctrs)})\n"
  if sum(cycle_ctrs) == 0:
   str_ccl += 'no odd convergence found\n'
  for i in range(len(ns_min)):
   str_ccl += f"{compls[i]:<12.6f} {frs[i]*100:<10.4f} {lens[i]:<8.0f}{ns_min[i]}\t{ns_max[i]}\n"
  print(str_ccl)  
 
 return results

'''
  #------------speed calculation, to determine if asymptotic stats recalc is needed------------
 
 #nips = collatz_speed_calc(mad, ('short', 'def'), n_digits=(100,200,500,1000,2000,5000), t_single=0.05)

 nips_cur = np.interp(nd_slope, nips['short'][0], nips['short'][1])
 n_iter_eff = np.interp(nd_slope, nips['short'][0], nips['short'][1]) * t_lim_slope
 print(md_slope, n_iter_eff)

'''

# slope_max, compl_max, jump_max = [vs[x] for x in ['slope_max', 'compl_max', 'jump_max']]
# slope_lim_inf, slope_lim_small = [vs[x] for x in ['slope_lim', 'slope_lim_small']]
# compl_lim, jump_lim = [vs[x] for x in ['compl_lim', 'jump_lim']]


def stats_to_str(vs, ftype=''): #(+.), add limits later
 '''
 Creates text string from stats for single Collatz parameters (dictionary, as outputted by stats_calc)
 '''


 mad = vs['mad']
 
 slope_avg, slope_std, compl_avg = [vs[x] for x in ['slope_avg', 'slope_std', 'compl_avg']]
 
 str_params_0 = f" {mad} "
 cycle_ctrs, div_ctr, undef_ctr = [vs[k] for k in ('cycle_ctrs', 'div_ctr', 'undef_ctr')]

 if sum(cycle_ctrs) == 0:
  str_params = '#=====' + str_params_0.ljust(32, '=')
  str_base = f" s:{slope_avg:.6f} c:{compl_avg:.6f} | no convergence found =====\n"
  str_wrt = str_params + str_base

 else:
  recs = vs['records']
  ((slope_max, compl_max, jump_max),(slope_max_num, compl_max_num, jump_max_num)) = recs.max_vals()
 
  str_header = ('\n#' + '='*30 + str_params_0).ljust(91, '=') + '\n\n'

  op_freq_str = '' #  op_freq_std_str = ''
  for i in range(len(vs['op_freqs'])):
   op_freq_str += f"{vs['op_freqs'][i]:.6f} "     #op_freq_std_str += f"{vs['op_freq_stds'][i]:.5f} "
 
  str_sc = f"Op_freqs: \t {op_freq_str} (std_max: {max(vs['op_freq_stds']):.6f})\n"
  str_sc += f"Avg slope (sigma), dispersion, compl \t{vs['slope_avg']:<10.6f} ({vs['slope_std']:.6f}) {vs['log_dispersion']:<7.4f} {vs['compl_avg']:.6f}\n"
  str_rec = f"Records (slope, compl, jump):\t{slope_max:.8f} {compl_max:.6f} {jump_max:.4f}\n" if slope_avg < 0 else ''
 # str_lims = f"SCJ lims, slope lim mid: \t{slope_lim_inf:<10.8f} {compl_lim:<10.8f} {jump_lim:<10.8f}; {slope_lim_small:<10.8f}\n"
  
  cycle_info = cycle_info_calc(vs)
  cycle_ctrs, ns_min, compls, frs, lens, ns_max = [cycle_info[k] for k in ('cycle_ctrs', 'cycle_ns_min', 'cycle_compls', 'cycle_freqs', 'cycle_lens', 'cycle_ns_max')]
  str_ccl = f"\nCycles compls, precentages, lengths, ns_min, ns_max:\n"
  for i in range(len(ns_min)): #zeroth is deleted by cycle_info
   n_max_str = str(ns_max[i]) if log10(ns_max[i]) < 60 else f"...............10**({log10(ns_max[i]):.8f})........................!"
   str_ccl += f"{compls[i]:<12.6f} {frs[i]*100:<10.4f} {lens[i]:<8.0f}{ns_min[i]}\t{n_max_str}\n"

  str_ccl += f"calculated total {sum(cycle_ctrs) + div_ctr + undef_ctr} (converged {sum(cycle_ctrs)}, diverged {div_ctr}, undefined {undef_ctr})\n"
  
  str_wrt = str_header + str_sc + str_rec + str_ccl + '\n'

 if 'print' in ftype:
  print(str_wrt)
    
 return str_wrt



def stats_write_all(data_all, fn='stats_txt.txt', ftype=''):
 '''
 Creates text string from stats for all Collatz parameters (dictionary of dictinaries outputted by stats_calc, with tuple keys containing multiplier, divisors and addition, e.g. (11, (2,3,5), 1) )
 '''
 str_wrt = ''
 for k in data_all.keys():
  str_cur = stats_to_str(data_all[k])
  str_wrt += str_cur

 wrt_type = 'w' if ('w' in ftype or fn not in os.listdir()) else 'a'
 if '.txt' in fn:
  with open(fn, wrt_type) as d:
   wrt = d.write(str_wrt)
 return str_wrt 
 
def mad_combos(n_primes=12, add_lim=102, n_divs_max=6, ftype='primes only'):
 '''
 Creates array of all (multiplier, addition, divisors) combinations for first 'n_primes' primes, number of prime divisors no more than n_divs_max and all possible additions up to add_lim
 '''
 
 prs = primes[:n_primes]
 ctr = 0
 mults_out, divs_out, adds_out = [], [], []
 adds_all = [1,] + ([x for x in primes if x < add_lim] if 'primes only' in ftype else [x for x in range(2, add_lim) if x % 2 != 0]) #if (x < add_lim and x != mult and not any([x == d for d in divs])) ]
 for i in range(1, len(prs)):
  mult = int(prs[i])
#  add_max = add_lim #, round(300 /mult))
  divs_all = [int(x) for x in prs[1:i]]
  div_list = []
  for n_divs in range(min(i, n_divs_max)): #25 for i == 6 and n_combo_max==4
   divs_cur = [(2,) + x for x in list(combinations(divs_all, n_divs))]
   div_list += divs_cur
  for divs in div_list:
   for add in adds_all:
    if any([add % x == 0 for x in divs + (mult,)]):
     continue
    mults_out.append(mult)
    adds_out.append(add)
    divs_out.append(tuple(divs))
    ctr += 1
 
 if 'arr' in ftype:
  return np.array((mults_out, adds_out, divs_out), dtype='object').T
 else:
  list_out = [ (mults_out[i], adds_out[i], divs_out[i]) for i in range(len(mults_out))]
  return list_out


def avg_slope_all(data_all={}, fn_d='', ks=None, n_digits=60, calc_len=122, n_iter_lim=1e9, f_cond=None, t_lim=2000,
                    exp_lim=2.0, sigma_thr=0.0, skip=51, ftype='recalc'):
 
 recalc_type = 'recalc' if 'recalc' in ftype else 'skip'
 print_type = 'print' if 'print' in ftype else ''
 return_dict = True if 'return' in ftype else False
 
 if fn_d != '' and fn_d in os.listdir():
  data_all = db_load(fn_d)
       
 if not (isinstance(ks, list) or isinstance(ks, tuple)):
  ks = db_keys(data_all)

 t_0 = time.time()
 ctr_mad = 0
 
 for mad in ks:
 
  n_digits_cur = n_digits[mad] if isinstance(n_digits, dict) and mad in n_digits.keys() else n_digits
  calc_len_cur = calc_len[mad] if isinstance(calc_len, dict) and mad in calc_len.keys() else calc_len
  n_iter_lim_cur = n_iter_lim[mad] if isinstance(n_iter_lim, dict) and mad in n_iter_lim.keys() else n_iter_lim
  t_lim_cur = t_lim[mad] if isinstance(t_lim, dict) and mad in t_lim.keys() else t_lim
  f_cond_cur = f_cond[mad] if isinstance(f_cond, dict) and mad in f_cond.keys() else f_cond
  exp_lim_cur = exp_lim[mad] if isinstance(exp_lim, dict) and mad in exp_lim.keys() else exp_lim
  sigma_thr_cur = sigma_thr[mad] if isinstance(sigma_thr, dict) and mad in sigma_thr.keys() else sigma_thr
  skip_cur = skip[mad] if isinstance(skip, dict) and mad in skip.keys() else skip
 
  if mad not in data_all.keys():
   data_all[mad] = {}
  
  data_present = val_check(data_all[mad], keys_inf, mad=mad) and data_all[mad]['len_block'] == calc_len_cur and data_all[mad]['t_calc_slope'] > t_lim_cur*0.7  
  
  if not data_present or recalc_type == 'recalc':   
  
   vs = avg_slope_calc(n_digits=n_digits_cur, calc_len=calc_len_cur, mad=mad, n_iter_lim=n_iter_lim_cur, ftype=print_type,
                      f_cond=f_cond_cur, t_lim=t_lim_cur, exp_lim=exp_lim_cur, sigma_thr=sigma_thr_cur, skip=skip_cur) 
   
   data_all[mad].update(vs)
   ctr_mad += 1 

   op_freq_str = ''
   op_freq_std_str = ''
   for i in range(len(vs['op_freqs'])):
    op_freq_str += f"{vs['op_freqs'][i]:.6f} "
    op_freq_std_str += f"{vs['op_freq_stds'][i]:.6f} "

   print(f"{time.time() - t_0:.1f} {ctr_mad} avg slope (sigma), compl, disp \t{vs['slope_avg']:<10.6f} ({vs['slope_std']:.6f}) {vs['compl_avg']:.6f} {vs['log_dispersion']:<7.4f} {mad}")
   print(f"op_freqs (sigmas) \t {op_freq_str} ({op_freq_std_str})")
 
  if sqrt(ctr_mad) % 1 == 0 and fn_d != '' and 'save' in ftype:
   db_save(data_all, fn_d)

 if fn_d != '' and 'save' in ftype:
  db_save(data_all, fn_d)
  
 return data_all if return_dict else None

  
def avg_slope_calc(mad=(), n_digits=60, calc_len=122, n_iter_lim=1e9, f_cond=None, t_lim=2000, n_bins=10,
                    exp_lim=2.0, sigma_thr=0.0, skip=51, ftype='', mult=3, add=1, divisors=[2,]):
 '''
 Calculates asymptotic statistics of Collatz sequence, defined by mult, divisors and addition.
 -Generates large random number in range (10**(n_digits), 10**(n_digits+1)), log-spaced
 -Iterates it skip times to avoid "start effects" (average factorization of random number is not equal to that of numbers in iteration log)
 -Iterates resulting number calc_len times, calculates integer-and float-type statistics on resulting sequence
 -Terminates if calc time or total iteration number limits are exceeded, or needed precision is reached (the latter specified by sigma_thr which is threshold of std dev of 5 subsample means of observed slopes (see sem_calc)
 -prints warnings if calc_len is too big (some sequences reach convergence or significant expansion)
 
 Takes:
  -n_digits - order of magnitude (number of digits) in which random number sample is generated
  -calc_len - length of iteration log for which statistics are calculated
  -skip - number of skipped iterations in the beginning of each log (to avoid 'start effect')
  --for each number, [skip:skip+calc_len] iteration log slice is evaluated
  -mad - (mult, add, divisors) tuple, in example (7,1,(2,5))
  -n_iter_lim - total number of iterations for which stats are calculated
  -t_lim - stop if t_lim elapsed (seconds)
  -exp_lim - expansion limit for Collatz iterations (log_n_max = n_digits*exp_lim)
  -f_cond: if specified as lambda function, includes only logs which pass under f_cond 
  --in example: completeness or slope in specified range: f_cond = lambda x: x[i_float][i_slope] > 0.01 on x as collatz_calc function output)
 
 Returns:
  Dictionary of values which include asymptotic slope, iteration log dispersion, operation type frequencies, their uncertainties, 
  histogram of slopes observed for subsequences of length calc_len, and empirical dependence of operation frequencies on local slope.
 
 if 'ol' in ftype, calculates full log up to (skip+calc_len) for each number; otherwise, uses short-type collatz function which does not calculate full logs (much faster)
 
 '''

 if isinstance(mad, tuple) and len(mad) == 3:
  mult, add, divisors = mad

 mad = (mult, add, divisors)
 op_types = ops(mad)
 drop_types = ops(mad, 'drop') #drops corresponding to op types
 
 #limit is defined by total number of collatz iterations, so thai it is constant if the slope is high and calc_len is small, or vice versa.
 op_ctrs_lst, sigmas, end_conds, ns = [], [], [], []
 cond, calc_ctr, iter_ctr = True, 0, 0
 t_0 = time.time()
 n_skip = int(calc_len*skip) if skip < 0 else int(skip)
 ol_return = 'ol' in ftype
 calc_drops = 'drop' in ftype
 collatz_ftype = 'def' if ol_return else 'short'
 ols, drops = [], []
 ec_wrong = 0
 
 valid_cond = ''
 while cond:
  t_cur = time.time() - t_0
  calc_ctr += 1
  n = range_gen(log_range=(n_digits, n_digits+1), n_sample=1, divisors=[])[0]
  if n_skip > 2: 
   vs = collatz_calc(n, mad=(mult, add, divisors), log_n_max=round(n_digits*exp_lim), ctr_lim=n_skip, ftype='short')
   n = vs[i_int][i_n_last] #skip first n_calcs ("start effect: in log, even number ratio is not 1/2")
  ns.append(n)
  vs = collatz_calc(n, mad=(mult, add, divisors), log_n_max=round(n_digits*exp_lim), ctr_lim=calc_len, ftype=collatz_ftype)
  vs_int, vs_float, iter_log, odd_log, num_buf = vs 
  if f_cond != None and not f_cond(vs_int, vs_float): 
   continue
  ctr, op_ctrs, n_first, n_last, jump, glide, end_cond = vs_int
  iter_ctr += ctr
  slope, compl, log_jump = vs_float
  op_ctrs_lst.append(op_ctrs)
  end_conds.append(end_cond)
  if ol_return:  
   ols.append(odd_log)
  drops.append(log10(n_last) - log10(n)) 
 
  if sqrt(calc_ctr) % 25 == 0 or iter_ctr >= n_iter_lim or t_cur > t_lim:
   op_ctrs_arr = np.array(op_ctrs_lst)
   op_freqs_arr = op_ctrs_arr / op_ctrs_arr.sum(axis=1)[:,None]
   slopes = (drop_types * op_freqs_arr).sum(axis=1)
   slope_mean, slope_sem = sem_calc(slopes) #n_rep=100, sub_len=100500
   op_ctrs_tot = op_ctrs_arr.sum(axis=0)
   op_freqs = op_ctrs_tot/op_ctrs_tot.sum()
   if 'print' in ftype:
    op_freq_str = ''
    for i in range(len(op_freqs)):
     op_freq_str += f"{op_freqs[i]:.6f} "
    print(f"{calc_ctr} {iter_ctr:.3g} {t_cur:.2f} {slope_mean:.6f} {slope_sem:.6f}   {op_freq_str}")
   if slope_sem < abs(slope_mean)*0.2 and slope_sem < sigma_thr: #5 sigma that sign of the mean is correct
    break
   ec_wrong = sum([x != 3 for x in end_conds])
   if ec_wrong / calc_ctr > 0.001 and 'decrease' not in valid_cond:
    valid_cond += 'decrease calc len'
    print(f"decrease calc_len: {ec_wrong} of {calc_ctr} sequences ended or diverged")
  if iter_ctr >= n_iter_lim or t_cur > t_lim:
   break
 
 compl_mean = op_freqs[0] / sum(op_freqs[1:]) 
 op_freq_stds = np.array([sem_calc(op_freqs_arr[:,i])[1] for i in range(len(mad[2])+1)])

 slope_arr = (op_freqs_arr*drop_types[None,:]).sum(axis=1)

 slope_bins = np.linspace(slope_arr.min(), slope_arr.max(), n_bins+1)
 slope_bin_means = np.zeros(n_bins)
 slope_hist = np.zeros(n_bins)
 op_freqs_sh = np.zeros((n_bins, drop_types.size))
 for i_bin in range(slope_bins.size - 1):
  slope_l, slope_r = slope_bins[i_bin], slope_bins[i_bin+1]
  is_slope = np.where((slope_l <= slope_arr) * (slope_arr < slope_r))[0] if i_bin != slope_bins.size - 2 else np.where((slope_l <= slope_arr) * (slope_arr <= slope_r))[0]
  slope_hist[i_bin] = is_slope.size / slope_arr.size
  slope_bin_means[i_bin] = slopes[is_slope].mean()
  op_freqs_bin_arr = op_freqs_arr[is_slope]
  op_freqs_bin = op_freqs_bin_arr.mean(axis=0)
  op_freqs_sh[i_bin] = op_freqs_bin / op_freqs_bin.sum()

 #calculating dispersion and non-gaussianity 
 drop_arr = np.sort(np.array(drops))
 drops_avg, drops_std = drop_arr.mean(), drop_arr.std() 
 drops_skew, drops_kur = scp.stats.skew(drop_arr), scp.stats.kurtosis(drop_arr)
 log_dispersion = drops_std / sqrt(calc_len)

 vs_out = {'mad': mad, 'slope_avg': slope_mean, 'compl_avg': compl_mean, 'log_dispersion': log_dispersion, 'slope_std': slope_sem, 'op_types': op_types, 'op_freqs': op_freqs, 'op_freq_stds': op_freq_stds, 
             'slope_bins': slope_bin_means, 'slope_hist': slope_hist, 'op_freqs_hist': op_freqs_sh}
 
 vs_out.update({'ols': np.array(ols), 'drops': (drop_arr if calc_drops else np.array([]))})
 vs_out.update({'drop_sk': drops_skew, 'drop_kur': drops_kur, 'len_block': calc_len, 'n_iter_tot': iter_ctr, 't_calc_slope': t_cur, 'ec_wrong_slopes': ec_wrong})
  
 return vs_out


def drop_calc(op_types, op_freqs, ftype=''):
 '''
 Calculates arrays of unique drop values and their frequencies from arrays of unique operation types and their frequencies
 op type: mult if mult+add, d if divided by d
 drop = log10(n_next) - log10(n) 
 '''
 drops_u = np.copy(op_types)
 drops_u = np.log10(drops_u)
 drops_u[1:] *= -1
 drop_freqs = np.copy(op_freqs)
 if 'raw' not in ftype:
  drops_u[0] += drops_u[1]
  drop_freqs[1] -= drop_freqs[0]
  drop_freqs = drop_freqs / drop_freqs.sum() #recalc OK
 return drops_u, drop_freqs

def ops(mad, ftype=''):
 '''
 Create array of op_types from (mult, add, divisors) tuple
 Example: (7, 1, (2,3,5)) -> np.array((7,2,3,5))
 '''
 mult, add, divisors = mad
 op_types = np.array((mult,) + tuple(divisors))
 if 'drop' in ftype:
  drop_types = np.copy(op_types).astype(float)
  drop_types[1:] = 1.0 / drop_types[1:]
  drop_types = np.log10(drop_types)
  return drop_types
 else:
  return op_types

#up to specified sigma in completeness, as in slope_inf_calc
def iter_stat_calc(mad, ols):
 ''' 
 Calculates statistics of iteration logs of Collatz function defined by mad=(multiplier, addition, divisors) argument
 -Generates random number witn nd digits (decimal)
 -Calculates iteration and operation type logs up to (skip + calc_len) iteration
 -Takes (skip:skip+calc_len) part of op type log
 -if f_cond function is specified, includes only logs which pass under f_cond (in example, completeness or slope in specified range: f_cond = lambda x: x[i_float][i_slope] > 0.01 on x as collatz_calc function output)
 -Repeats until total number of iterations n_iter_lim is reached or t_lim elapsed
 -Calculates frequencies of each type of operations in iteration logs
 
 Returns:
 list of operation type logs, array of unique operation types, array of operation type frequencies, uncertainties of frequencies, total number of iterations on which statistics are calculated, average odd/even ration for these iterations
 '''
 ol_arr = np.array([item for sublist in ols for item in sublist]) if isinstance(ols[0], list) else np.array(ols)
 op_types = ops(mad)
 op_freqs = np.zeros_like(op_types).astype(float)
 len_tot = ol_arr.size
 for i in range(len(op_types)):
  op_type = op_types[i] if op_types[i] != mad[0] else 0
  op_freqs[i] = np.where(ol_arr == op_type)[0].size / len_tot
 
 return op_types, op_freqs

def op_freq_to_cont_frac(op_freqs, thres=40):
 n_vals = op_freqs.size
 cf_arr = np.zeros((n_vals, 3), dtype=int)
 for i_op in range(n_vals):
  cfs = cont_frac(op_freqs[i_op], ftype='all')
  i_best = np.where(cfs[2] > thres)[0][0] if cfs[2].max() > thres else np.argmax(cfs[2])
  cf_arr[i_op] = cfs[:, i_best]
 return cf_arr

def op_freq_calc(mad=(), log_range=(12,12.5), n_sample=1e6, nums=None, return_nums = True, n_iter=1, ftype=''):
 '''
 The same as iter_stat_calc, but using Collatz iteration on numpy arrays of integers
 Faster than iter_stat_calc, but less robust and precise because numbera are limited by long long int (2**64-1)
 '''

 if n_iter > 1:
  while n_iter > 0:
   op_types, op_freqs, nums = op_freq_calc(mad, n_sample=n_sample, log_range=log_range, nums=nums, return_nums=True, n_iter=1)
   n_iter -= 1
   if 'print' in ftype:
    print(op_freqs, log10(max(nums)), log10(min(nums)))
  return op_freqs
  
 mult, add, divisors = mad
 divprod = prod(divisors)
 op_types = ops(mad)
 op_counts = np.zeros(len(divisors)+1, dtype=int)
 
 if not isinstance(nums, np.ndarray): 
  nums_0 = np.random.randint(int(10**log_range[0]), int(10**log_range[1]), int(n_sample), dtype=np.int64)  #range_gen((12.0,12.5), n_sample=100, divisors=divisors)
  nums_odd = nums_0[np.where(np.gcd(nums_0, divprod)==1)]
  nums = nums_odd * mult + add

 op_counts[0] += nums.size
 for i_d in range(len(divisors)):
  d = divisors[i_d]
  while True:
   mask_cur = np.where(nums % d == 0)[0]
   mask_size = mask_cur.size
   op_counts[i_d + 1] += mask_size
   if mask_size == 0:
    break 
   nums[mask_cur] = nums[mask_cur] / d

 op_freqs = op_counts / op_counts.sum()
 
 if return_nums:
  return op_types, op_freqs, nums * mult + add
 else:
  return op_types, op_freqs


def drop_stats_merge(dr_vals, dr_cts, n_mult=2, n_bins=100500, n_tails=0, nd_round=3, ftype=''):
 '''
 merge histograms of unique values dr_vals and their counts dr_cts
 '''
 dr_raw = np.repeat(dr_vals, dr_cts)
 n_dr_all = dr_raw.size
 n_dr_new = int(n_dr_all / n_mult)
 n_dr_int = n_dr_new * n_mult
 dr_arr = np.random.permutation(dr_raw[:n_dr_int]).reshape((n_dr_new, n_mult))
 drops_new = dr_arr.sum(axis=1)
 dr_vals_new, dr_cts_new = hist_with_means(drops_new, n_bins=n_bins, n_tails=n_tails, nd_round=nd_round, ftype=ftype)
 return dr_vals_new, dr_cts_new

#remaining
def ol_stats(ols, len_block=102):
 '''
 Makes avg_slope_calc-like dictionary of statistical values from list of op-type logs
 '''
 if not isinstance(ols, np.ndarrray):
  ol_arr = np.array([item for sublist in ols for item in sublist])
 else:
  ol_arr = ols
 
 len_ols, n_iters =  ol_arr.shape[1], ol_arr.size
 
 ols_fl = ol_arr.flatten()
  
 inds_st = np.sort(np.random.randint(0, len_all-len_block, n_blocks))
 inds_arr = np.array([np.arange(i, i+len_block) for i in inds_st])
 drop_arr_2d = drop_arr[inds_arr]
 drops = drop_arr_2d.sum(axis=1)
 drop_vals, drop_cts = hist_with_means(drops, n_bins=100500, n_tails=0, nd_round=4, ftype='')
 drop_wts = drop_cts / drop_cts.sum()
 slope_cur = (drop_wts*drop_vals).sum()/len_block


def slope_compl_calc(slope=nan, compl=nan, 
                     slope_bins=None, op_freqs_hist=None, compl_hist=None, 
                     op_freqs=None, op_types=None, mad=None, ftype=''):
 '''
 Calculates completeness from slope and vice-versa:
 Calculation types by ftype strings and arguments:
 -'classic' ftype: 
 --calculate by theoretical formula ('asymptotically strict' slope-compl relation if number of divisors is 1)
 
 -if op_freqs and op_types are specified: 
 --calculate with tilted probabilities, 
 ---if op_types and op_freqs are provided, single-iteration slope is used
 ---if effective drop_types and their frequencies are provided (where mult/div2 is treated as single iteration), 
 ----slope value per effective iteration is calculated
 ----which is recalculated to single-iteration value if 'drop' in ftype
 
 -if slope_bins and op_freqs_hist are provided:
 --perform direct calculation, using interpolation
 
 '''
 if not (isfinite(compl) or isfinite(slope)):
  return nan
 
 ret_type = 'compl' if isfinite(slope) else 'slope'
 
 calc_type = ''
 if 'classic' in ftype:
  calc_type = 'classic' 
 elif  not (op_types is None or op_freqs is None):
  calc_type = 'single'
 elif not (slope_bins is None or op_freqs_hist is None):
  calc_type = 'block'
  
 if 'block' in calc_type:
  compl_hist = op_freqs_hist[:,0] / op_freqs_hist[:,1:].sum(axis=1)
  if isfinite(slope): #calculate completeness #np.interp(val, slope_bins, compl_hist)
   f_interp = scp.interpolate.interp1d(slope_bins, compl_hist, kind='cubic', fill_value='extrapolate')
   compl = f_interp(slope)
  else: #calculate slope
   f_interp = scp.interpolate.interp1d(compl_hist, slope_bins, kind='cubic', fill_value='extrapolate')
   slope = f_interp(compl)

 elif 'single' in calc_type: #single iteration-type calc, with tilted frequencies
  (drops_u, drop_freqs) = drop_calc(op_types, op_freqs, ftype='raw') if 'ops' in ftype else (op_types, op_freqs)
  if isfinite(slope): #calculate completeness
   probs = tilted_prob_calc(slope, drops_u, drop_freqs)
   compl = probs[0] if 'ops' not in ftype else probs[0] / probs[1:].sum()
  else:  #calculate slope    
   if 'ops' in ftype:
    def f_sl(slope, compl=compl, drops_s=drops_u, op_freqs=op_freqs):
     probs_tilt = tilted_prob_calc(slope, drops_s, op_freqs)
     compl_tilt = probs_tilt[0] / probs_tilt[1:].sum()
     return compl_tilt - compl
    slope = scp.optimize.newton(f_sl, x0=0.0)
   else:
    f_sl = lambda x: tilted_prob_calc(x, drops_u, drop_freqs)[0] - compl #compl_avg = op_freqs[0] / op_freqs[1:].sum()
    slope_eff = scp.optimize.newton(f_sl, x0=0.0)
    slope = slope_eff / (1 + compl)

 elif 'classic' in calc_type: 
  mult = mad[0] if not mad is None else 3
  if isfinite(slope): #calculate completeness
   compl = (log10(2) + slope) / (log10(mult) - slope) 
  else:  #calculate slope
   slope = (compl*log10(mult) - log10(2)) / (1 + compl)
 
 else: 
  slope = compl = nan
 
 return float(compl) if ret_type == 'compl' else float(slope)
 
  
'''
   def f_sl(x, drops_u=drops_u, drop_freqs=drop_freqs, compl=compl):
    probs_t = tilted_prob_calc(x, drops_u, drop_freqs)
    compl_cur = probs_t[-1] / probs_t[:-1].sum()
    return compl_cur - compl

'''


def ctr_max_jump_calc(jump_func, jump, ctr_min, tol=1e-3, step=1.5):
 '''
 See limits_calc docstring
 '''
 ctr_start = ctr_min*(step - (step-1)/2)
 ctrs, probs = [ctr_start,], [jump_func(ctr_start, jump),]
 
 while (ctrs[-1] < 1e9 and ctrs[-1] > 0):
  ctrs.append(ctrs[-1]*step)
  probs.append(jump_func(ctrs[-1], jump))
  if len(ctrs) > 2 and isfinite(sum(probs[-3:])) and probs[-1] < probs[-2]: #extremum location
   break  

 i_start = np.where(np.isfinite(probs))[0][0]
 i_max = np.nanargmax(probs)
 
 if i_max == i_start:
  ctrs.append(ctrs[i_max])
  probs.append(probs[i_max])
  return ctrs, probs

 while abs(ctrs[-1] - ctrs[-2]) / ctrs[-1] > tol and abs(ctrs[-1] - ctrs[-3]) / ctrs[-1] > tol:
  ctr_xtr = parabola_vertex_calc(ctrs[-3:], probs[-3:])
  prob = jump_func(ctr_xtr, jump)
  if ctr_xtr < 0 or not isfinite(prob):
   ctrs.append(ctrs[i_max])
   probs.append(probs[i_max])
   break  
  ctrs.append(ctr_xtr)
  probs.append(prob)
  
 return ctrs, probs


#ftype: 
#'raw drops': calculate by values for single iteration; 
#'rate_func' - calculate with old-type probability calc (used for single drop calc)
#'' - calculate with fixed block length
#if odd logs are specified in ols arg, calculate importance sampling (very slow, seems to converge with block-type calculation
#dr_vals=None, dr_cts=None, op_types=None, op_freqs=None, ols=None, calc_len=51

def limits_calc(mad=(3,1,(2,)), log_num=1000.0, stats_inf={}, step=1.2,
                n_iter_lim=1e9, n_digits=120, t_lim=10, ftype=''):
 '''
 Calculates limits of slope and jump values for converging-type Collatz functions, based on observed/theoretical iteration statistics:
 Takes: 
 -(multiplier, addition, (divisors)) tuple, 
 -Order of magnitude of a starting number for which limits are calculated
 Additional arguments:
 -n_iter_lim, calc_len, n_digits, t_lim - see avg_slope_calc docstring
 -dr_vals, dr_cts - values and counts (histogram) of (log(n_end)-log(n_start)) after calc_len iterations
 -op_types, op_freqs - operation types and their frequencies (observed and theoretical)
 -ols = list of operation type logs (unnecessary)
 
 slope = log(n_end) - log(n_start) / number of iterations
 jump = log(max(iteration log)) / log(n_start)
 drop = (log(n_end) - log(n_start) 
 
 Uses assumption that any iteration log is a Markov chain: 
 -each (mult+add) iteration "resets" factorization of a resulting number on the set of divisors.
 -Thus, in any descending sequence which consists of consecutive division steps bounded by mult+add steps, 
 --numbers of divisions of each type is independent from these numbers in previous descending sequence and all previous iteration log, 
 --or at least, there are no persisting correlations.
 -Each Collatz function is characterized by well-defined frequencies, with which operation types appear in iteration logs, and their distributions
 
 Under this assumption:
 -there exists a probability that each iteration, or a block of iterations, will have a slope > some threshold
 -probability that a long iteration log consisting of N iterations or their blocks will have slope > threshold, is something like exp(-N*prob_threshold)
 -The close is the slope to zero for some number under a converging-type Collatz function, the lower is prob_threshold and simultaneously, the bigger is N (the longer it takes to converge, having this slope)
 -Thus, probability that a number will have iteration log longer than N, decreases faster than exp**(-N).
 -Thus, for each order of magnitude, there exist a limiting slope for which this probability is <= 1/num (1 / 10**(order_of_magnitude)
 
 The same logic is applied to jump limit, but here, a limiting value of jump is calculated for which for any possible sequence lengths, probability of such log length is <= 1 / num.
 -The higher is the slope, the lower is probability that a single iteration/block will have such slope
 -The lower is the slope, the greater number of iterations is needed to reach 10**(log(n_start) * jump) with this slope
 -Thus, for any given value of jump, probability that it will take N iterations/blocks to reach maximum value, vs N, firstly increases and then decreases.
 --this maximum is calculated with iterative search by ctr_max_jump_calc function
 -Jump limit is the value of log_jump for which max(probability) = 1 / num.
 -Probabilities that greater maximum could be reached with same number of iterations and higher slope, or with given slope and bigger number of iterations, drop sharply, so jump limit which is calculated this way should approach real value asymptotically with log(n_start) -> inf.
 '''
 
 #----------------adding needed values------------
 
 slope_avg = stats_inf['slope_avg'] if 'slope_avg' in stats_inf.keys() else None
 compl_avg = stats_inf['compl_avg'] if 'compl_avg' in stats_inf.keys() else None
 slope_bins = stats_inf['slope_bins'] if 'slope_bins' in stats_inf.keys() else None
 slope_hist = stats_inf['slope_hist'] if 'slope_hist' in stats_inf.keys() else None
 op_types = stats_inf['op_types'] if 'op_types' in stats_inf.keys() else None
 op_freqs = stats_inf['op_freqs'] if 'op_freqs' in stats_inf.keys() else None
 len_block = stats_inf['len_block'] if 'len_block' in stats_inf.keys() else None
 ols = stats_inf['ols'] if 'ols' in stats_inf.keys() else None
 drops = stats_inf['drops'] if 'drops' in stats_inf.keys() else None
 op_freqs_hist = stats_inf['op_freqs_hist'] if 'op_freqs_hist' in stats_inf.keys() else None
 log_dispersion = stats_inf['log_dispersion'] if 'log_dispersion' in stats_inf.keys() else None
 drop_sk = stats_inf['drop_sk'] if 'drop_sk' in stats_inf.keys() else 0.0
 drop_kur = stats_inf['drop_kur'] if 'drop_kur' in stats_inf.keys() else 0.0
 drops = stats_inf['drops'] if 'drops' in stats_inf.keys() else None

 drop_vals = (slope_bins * len_block) if not (slope_bins is None or len_block is None) else None
 drop_wts = slope_hist / slope_hist.sum() if not(slope_hist is None) else None
 
#dr_vals=None, dr_cts=None, op_types=None, op_freqs=None, ols=None, calc_len=51 
 
 #------------------recognizing calculation type, checking needed data--------------
  
 calc_type = 'imp' if 'imp' in ftype else ('single' if 'single' in ftype else 'block')
 
 if 'block' in ftype:
  if 'saddle' in ftype:
   calc_type += ' saddle'
  elif 'gauss' in ftype:
   calc_type += ' gauss'
  elif 'raw' in ftype and not (drops is None):
   calc_type += ' raw'
  else:
   calc_type += ' rate'

 if 'print' in ftype:
  print(f"calc_type: {calc_type}")
  
 if 'raw' in calc_type:
  drop_vals = drops
  drop_wts = np.ones_like(drops) / drops.size

 avg_vals_needed = (slope_avg is None or compl_avg is None)
 drop_stats_needed = ('block' in calc_type and (drop_vals is None or drop_wts is None or op_freqs_hist is None or len_block is None or log_dispersion is None) )
 op_stats_needed = (op_types is None or op_freqs is None)
 ols_needed = ('imp' in calc_type and ols is None)

 stat_calc_needed = drop_stats_needed or op_stats_needed or ols_needed or avg_vals_needed  
 #drop_stats_needed, op_stats_needed, ols_needed, avg_vals_needed  
 
 #-------------------preliminary calculations: exit if average slope > 0 (records are infinite)
 
 if avg_vals_needed:
  stats_prelim = avg_slope_calc(mad=mad, len_block=len_block, n_iter_lim=n_iter_lim, n_digits=n_digits, t_lim=t_lim/10, ftype='')
  slope_avg, compl_avg = stats_prelim['slope_avg'], stats_prelim['compl_avg']
 if slope_avg > 0:
  return slope_avg, compl_avg, inf, slope_avg #0.0 in place of average completeness, fix it later
 
 #-----------------calculating needed values
 if stat_calc_needed:
  print(f"drop_stats_needed: {drop_stats_needed}, op_stats_needed: {op_stats_needed}, ols_needed: {ols_needed}, avg_vals_needed: {avg_vals_needed}; calculating stats")
  sl_ftype = 'drops ' + ('ols' if ols_needed else '')
  nd = n_digits if 'imp' not in ftype else 1000
  lb = len_block if 'imp' not in ftype else min(3000, int(abs((0.3 * 1000 / slope_avg))))
  stats_inf = avg_slope_calc(mad=mad, calc_len=len_block, n_iter_lim=1e9, n_digits=120, t_lim=t_lim, ftype='')
 if drop_stats_needed:
  slope_bins, slope_hist, op_freqs_hist, len_block, log_dispersion = [stats_inf[x] for x in ['slope_bins', 'slope_hist', 'op_freqs_hist', 'len_block', 'log_dispersion']]
  #add skewness and kurtosis here
  drop_vals, drop_wts = slope_bins * len_block, slope_hist
 (op_types, op_freqs) = [stats_inf[x] for x in ['op_types', 'op_freqs']] if op_stats_needed else (op_types, op_freqs)
 ols = stats_inf['ols'] if ols_needed else ols
 
 if 'single' in calc_type:
  drops_u, drop_freqs = drop_calc(op_types, op_freqs) 
 
 
 #--------------slope limit calculation--------------
 if calc_type == 'imp':
  ctr_min, ctr_cur = -log_num/drops.min()*len_block, -log_num / slope_avg
 elif 'block' in calc_type:
  if 'gauss' in ftype:
   ctr_min = ctr_cur = -log_num / slope_avg
  else:
   ctr_min, ctr_cur = -log_num / drop_vals.min() * len_block, -log_num / slope_avg
 else:
  ctr_min, ctr_cur = -log_num / drops_u.min(), -log_num / slope_avg

 if calc_type == 'imp':
  prob_func = lambda x: log_num + prob_calc_is(log_n_start=log_num, log_n_end=0.0, ctr=x, drop_arr=drop_arr, mad=mad)
 elif calc_type == 'single': 
  prob_func = lambda x: log_num + prob_calc(log_num, 0.0*log_num, x, drops_u=drops_u, drop_freqs=drop_freqs) 
 elif 'rate' in calc_type:
  prob_func = lambda x: log_num + prob_calc(log_num, 0.0, x/len_block, drops_u=drop_vals, drop_freqs=drop_wts, ftype='log') #x = n_iter/len_block
 elif 'gauss' in calc_type:
  prob_func = lambda x: log_num + prob_calc_gauss(log_num, 0.0, x, len_block=len_block, slope_avg=slope_avg, log_dispersion=log_dispersion, drop_sk=drop_sk*0.0, drop_kur=drop_kur*0.0) #x = n_iter/len_block
 else: #saddle pt
  prob_func = lambda x: log_num + prob_calc_block(log_n_start=log_num, log_n_end=0.0, ctr=x, len_block=len_block, drop_hist=drop_vals, drop_weights=drop_wts)

 ctrs, probs= [ctr_cur,], [log_num,]
 while probs[-1] > 0: #coarse: determine bin search interval
  ctr_cur *= step
  prob = prob_func(ctr_cur)
  ctrs.append(ctr_cur)
  probs.append(prob)

 ctr_max = bin_search(prob_func, 0, x_range=(ctr_cur/2, ctr_cur)) #fine: calculate precise value with binary search
 if calc_type == 'single':
  drop_eff = -log_num / ctr_max
  compl_lim = tilted_prob_calc(drop_eff, drops_u, drop_freqs)[0]
  slope_lim = drop_eff / (1 + compl_lim)
 else:
  slope_lim = -log_num / ctr_max
  compl_lim = slope_compl_calc(slope=slope_lim, slope_bins=slope_bins, op_freqs_hist=op_freqs_hist)

 #---------------------max jump calculation
 if not ('single' in ftype or 'gauss' in ftype) and drop_vals.max() < 0: #slope before jump is positive -> cannot be calculated
  return slope_lim, compl_lim, nan, nan  

 if calc_type == 'imp':
  jump_func = lambda x, y: log_num + prob_calc_is(log_n_start=log_num, log_n_end=log_num*y, ctr=x, drop_arr=drop_arr, mad=mad)
 elif calc_type == 'single':
  jump_func = lambda x, y: log_num + prob_calc(log_num, log_num*y, x, drops_u=drops_u, drop_freqs=drop_freqs) 
 elif 'rate' in calc_type:
  jump_func = lambda x, y: log_num + prob_calc(log_num, log_num*y, x/len_block, drops_u=drop_vals, drop_freqs=drop_wts, ftype='log')
 elif 'gauss' in calc_type:
  jump_func = lambda x, y: log_num + prob_calc_gauss(log_num, log_num*y, x, len_block=len_block, slope_avg=slope_avg, log_dispersion=log_dispersion, drop_sk=drop_sk*0.0, drop_kur=drop_kur*0.0) #x = n_iter/len_block
 else: #saddle pt calculation
  jump_func = lambda x, y: log_num + prob_calc_block(log_n_start=log_num, log_n_end=log_num*y, ctr=x, len_block=len_block, drop_hist=drop_vals, drop_weights=drop_wts)
 
 
 ctrs_max, probs_max, jumps = [], [], [1.2,] #jump is never less than 1.4
 drop_lim = log10(mad[0] / mad[2][0]) if ('single' in calc_type or 'gauss' in calc_type) else drop_vals.max()
 #ctr_mult = len_block/drop_vals.max() if calc_type != 'single' else 2.0/drops_u[-1]

 while True:
  ctr_j_min = (log_num * (jumps[-1] - 1) / drop_lim) * 1.01
  ctrs_j, probs_j = ctr_max_jump_calc(jump_func, jumps[-1], ctr_j_min, tol=1e-3, step=step)
  ctr_j_max, prob_j_max = ctrs_j[-1], probs_j[-1]
  ctrs_max.append(ctr_j_max)
  probs_max.append(prob_j_max)
  if (prob_j_max < 0 or isnan(prob_j_max) or jumps[-1] > 100500):
   break
  jumps.append(jumps[-1] * step)
 
 i_last = min(np.where(np.array(probs_max) > 0)[0][-1], len(jumps)-2)
 if len(jumps) == 1 or jumps[-1] > 100500 or np.nanmax(probs_max[i_last + 1]) > 0:
  jump_lim, ctr_jump_max = jumps[0], ctrs_max[0]
 else:
  jump_lim_func = lambda x: ctr_max_jump_calc(jump_func, x, (log_num * (jumps[-1] - 1) / drop_lim) * 1.01, tol=1e-3, step=1.2)[1][-1]
  jump_lim = bin_search(jump_lim_func, 0, x_range=(jumps[i_last], jumps[i_last + 1]), prec=1e-3)
  ctrs_jump_max, probs_jump_max = ctr_max_jump_calc(jump_func, jump_lim, (log_num * (jump_lim - 1) / drop_lim) * 1.01, tol=1e-3, step=1.2)
  ctr_jump_max, prob_jump_max = ctrs_jump_max[-1], probs_jump_max[-1]
 
 if 'viz' in ftype:
  ctrs = np.logspace(log10(ctr_min)+0.1, log10(ctr_min) + 10.1, 101)
  jumps = np.hstack(( np.logspace(0.1, 2.1, 61), np.logspace(2.1, 6.1, 41)[1:] ))
  n_ctrs, n_jumps = ctrs.size, jumps.size
  probs_arr = np.ones((jumps.size, ctrs.size)) * nan
  cont1, cont2 = True, True
  i_j = -1
  for i_j in range(jumps.size):
   for i_ctr in range(ctrs.size):
    probs_arr[i_j, i_ctr] = jump_func(ctrs[i_ctr], jumps[i_j])
   if i_j > 3 and np.nanmax(probs_arr[i_j-2]) < 0:
    break

  probs_arr[np.where(np.isnan(probs_arr))] = -inf

  plt.clf()
  for i in range(probs_arr.shape[0]):
   p=plt.plot(np.log10(ctrs), probs_arr[i]/log_num)

  i_ctrs_max = np.nanargmax(probs_arr, axis=1)
  ctrs_max = ctrs[i_ctrs_max]
  plt.grid(True)
  plt.ylim(-2, 2)
  plt.xlim(log10(ctr_min)*0.8, log10(ctrs_max[i_j-2])*2.0)

  i_ctr_viz = np.where((ctrs_max > 0) * (np.isfinite(ctrs_max)))[0]
  #plt.plot(np.log10(ctrs_max[i_ctr_viz]), probs_arr[i_ctr_viz])
  print(jumps[:i_j])
 
 if 'single' in ftype:
  jump_slope_eff = log_num * (jump_lim - 1) / ctr_jump_max
  compl_max_jump = tilted_prob_calc(jump_slope_eff, drops_u, drop_freqs)[0]
  jump_slope = jump_slope_eff / (1 + compl_max_jump)
 else:
  jump_slope = log_num * (jump_lim - 1) / ctr_jump_max
 
 return slope_lim, compl_lim, jump_lim, jump_slope 


def cycle_calc_all(db=None, fn_d='', ftype=''): #1s on 1484 keys with high cycles
 
 if fn_d in os.listdir():
  with open(fn_d, 'rb') as d:
   db = pickle.load(d)
 
 ks = db_keys(db, ftype='', cond='')
 for mad in ks:
  cycles = [[0,]]
  for n_min in db[mad]['cycle_ns_min'][1:]:
   il = collatz_calc(n_min, mad=mad, log_n_max=10000, ctr_lim=3e8, ftype='def')[i_il]
   cycles.append(il[:-1])
  db[mad]['cycles'] = cycles
 
 if 'save' in ftype and fn_d != '':
  with open(fn_d, 'wb') as d:
   pickle.dump(db, d)

 
 return None


def cycle_info_calc(vs, ftype=''):
 mad, ns_min_raw = (vs[x] for x in ('mad', 'cycle_ns_min'))
 cycle_ctrs_raw = vs['cycle_ctrs'] if 'cycle_ctrs' in vs.keys() else [1 for n in ns_min_raw]
 div_ctr = vs['div_ctr'] if 'div_ctr' in vs.keys() else 0
 undef_ctr = vs['undef_ctr'] if 'undef_ctr' in vs.keys() else 0
 
 i_srt = [i for i, _ in sorted(enumerate(ns_min_raw), key=lambda x: x[1])]
 ns_min = [ns_min_raw[i] for i in i_srt]
 cycle_ctrs = [cycle_ctrs_raw[i] for i in i_srt]

 if ns_min[0] == 0:
  ns_min, cycle_ctrs = ns_min[1:], cycle_ctrs[1:]
 
 ns_max, cycle_lens, cycle_compls, cycle_op_freqs, cycles = [], [], [], [], []
 for i_c in range(len(ns_min)):
  n_min = ns_min[i_c]
  vs_ccl = collatz_calc(n_min, mad=mad, ctr_lim=3e8, log_n_max=int(1e5), ftype='def')
  ccl = vs_ccl[i_il][:-1]
  cycle_len, n_max = len(ccl), max(ccl)
  compl = vs_ccl[i_float][i_compl]
  op_ctrs = vs_ccl[i_int][i_ctrs]
  op_freqs = [x/cycle_len for x in op_ctrs]
  ns_max.append(n_max)
  cycle_lens.append(cycle_len)
  cycle_compls.append(compl)
  cycle_op_freqs.append(op_freqs)
  if 'full' in ftype:
   cycles.append(ccl)
 
 ctr_tot = sum(cycle_ctrs)
 cycle_freqs = [x/ctr_tot for x in cycle_ctrs]
 cycle_freqs_abs = [x/(ctr_tot + div_ctr + undef_ctr) for x in cycle_ctrs]
 
 f_arr = (lambda x: np.array(x)) if 'arr' in ftype else (lambda x: x)
 
 vs_out = { 'cycle_ns_min': f_arr(ns_min), 'cycle_ns_max': f_arr(ns_max), 'cycle_lens': f_arr(cycle_lens), 'cycle_ctrs': cycle_ctrs, 'cycle_freqs': f_arr(cycle_freqs), 'cycle_freqs_abs': f_arr(cycle_freqs_abs),
           'cycle_compls': f_arr(cycle_compls), 'cycle_op_freqs': f_arr(cycle_op_freqs), 'cycles': cycles}
 
 return vs_out

'''
 
 #cycles_srt = [cycles[i] for i in i_srt]
 cycle_ctrs_srt = [cycle_ctrs[i] for i in i_srt] + [div_ctr, undef_ctr]
 ns_min_srt = [ns_min[i] for i in i_srt]
 ns_max_srt = [ns_max[i] for i in i_srt]
 cycle_compls_srt = [cycle_compls[i] for i in i_srt]
 cycle_lens_srt = [cycle_lens[i] for i in i_srt]
 
'''
 	 
def collatz_viz(ns = [], mad=(3, 1, (2,)), ctr_lim=1e7, log_n_max=1000, log_slice=100, ctrs=None,
             n_digits=100, i_start=0, n_sample=100, figax=None,
             x_lim=None, y_lim=None, xscale='log', y_scale='lin', clr=None, alpha=0.5, ftype=''):
 
 '''
 ftype:
 'odd' - only odd starting numbers
 'slope' - cumulative slopes
 'log' - visualize log10(iter_log)
 'log_odd' - visualize only odd iterations
 '''
 if isinstance(mad, tuple) and len(mad) == 3:
  mult, add, divisors = mad
 else:
  mult, add, divisors = 3, 1, [2,]
 
    
 logs_all = [] #logarithms of numbers
 ols_all = []
 slopes_all = []
 compls_all = []
 
 odd_nums = 'odd' in ftype
 odd_log = log_slice == -1
 if log_slice < -1:
  log_slice = 1
  
 viz_type = 'log' if 'log' in ftype else ('compl' if 'compl' in ftype else 'slope')
 
 ctrs_present = isinstance(ctrs, np.ndarray)
 if not ctrs_present:
  ctrs = np.arange(0, int(ctr_lim) + 1, log_slice)
 
 if len(ns) == 0:
  ns = range_gen(log_range=(n_digits, n_digits + 1), n_sample=n_sample, divisors=(divisors if odd_nums else []))
 len_sample = len(ns)
 
 for i_n in range(len_sample):
  n_0 = ns[i_n]
  if log_n_max < log10(n_0):
   log_n_max = int(log10(n_0) * 2)
  if log_slice < 100 and not ctrs_present: 
   vs = collatz_calc(n_0, mad=(mult,add,divisors), log_n_max=log_n_max, ctr_lim=ctr_lim, log_slice=log_slice, ftype='py_short')
   vs_int, vs_float, iter_log, odd_log, num_buf = vs 
   ctr, op_ctrs, n_first, n_last, jump, glide, end_cond = vs_int
   slope, compl, log_jump = vs_float
   if odd_log:
    iter_log = [iter_log[i] for i in range(len(iter_log)) if odd_log[i] == 0]
   
  else:  #several times faster
   iter_log, ctr = [n_0,], 0
   n = n_0
   for i in range(1, len(ctrs)):
    vs_i = collatz_c_short(n, mult=mult, add=add, divisors=divisors, log_n_max=log_n_max, ctr_lim=ctrs[i] - ctrs[i-1])
    n = vs_i[i_n_last] 
    iter_log.append(n)
    if vs_i[-1] != 3: # log_n_max reached or cycle is entered
     break
   
  logs_all.append([log10(x) for x in iter_log])
  
 if 'clf' in ftype:
  plt.close('all')

 fig, ax = plt.subplots() if figax == None else figax 
 
 for i_n in range(len_sample):
  log_cur = logs_all[i_n]
  if viz_type == 'slope':   #sl_cum = np.cumsum(np.diff(il_arr) / np.arange(1, il_arr.size)) #completeness
   sl_cum = [(log_cur[i] - log_cur[0]) / (ctrs[i] - ctrs[0]) for i in range(1, len(log_cur))]
   p = ax.plot(ctrs[:len(sl_cum)], sl_cum, c=clr, alpha=alpha) #OK
  elif viz_type == 'log':
   p = ax.plot(ctrs[:len(log_cur)], log_cur, c=clr, alpha=alpha)
 
 if isinstance(y_lim, tuple):  	
  plt.ylim(y_lim)
 if isinstance(x_lim, tuple):
  plt.xlim(x_lim)
 plt.grid(True)

 return fig, ax


from collatz_viz import *


rec_compl = []
with open('nums_compl.txt', 'r') as d:
 ls = d.readlines()

for i in range(len(ls)):
 s = ls[i] 
 n_str = s.split('\t')
 if len(n_str) > 1:
  rec_compl.append(int(n_str[1])) #records_compl.append(to_arr_large(int(n_str[1])))


rec_glide = []
with open('nums_glide.txt', 'r') as d:
 ls = d.readlines()

for i in range(len(ls)):
 s = ls[i] 
 n_str = s.split('\t')
 if len(n_str) > 2:
  rec_glide.append(int(n_str[2]))


rec_jump = []
with open('nums_jump.txt', 'r') as d:
 ls = d.readlines()

for i in range(len(ls)):
 s = ls[i] 
 n_str = s.split('\t')
 if len(n_str) > 2:
  rec_jump.append(int(n_str[1]))


#unnecessary functions, older/unoptimal versions, ...

'''

'''
