import numpy as np
Poly = np.polynomial.polynomial.Polynomial
import scipy as scp
from numba import jit, njit
from numba.typed import List
from math import *
import random
import cmath
from decimal import *
from mpmath import mp, mpf, mpc
from mpmath import rand as mp_random
mp.dps = 1000
import gmpy2
from gmpy2 import mpz, is_prime, divexact, powmod, gcd

import warnings
from itertools import combinations
import copy
import bisect
import bitarray as ba
import pdb
import timeit
from matplotlib import pyplot as plt
from pprint import pprint as pp
import time
import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.markers import MarkerStyle
import matplotlib.colors as colors
import matplotlib.projections as projections
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

import PIL

from numba import config
config.DISABLE_JIT = False


arr_dummy = np.empty(0) 
ns_min_def = [0,]
zs_min_def = ns_min_def_arr = np.array(ns_min_def)
div_def = [2,]
div_def_arr = np.array(div_def)


i_int, i_float, i_il, i_ol, i_res_buf = 0, 1, 2, 3, 4
i_ctr, i_ctrs, i_n_start, i_n_last, i_jump, i_glide, i_end_cond = [0,1,2,3,4,5,6]
i_slope, i_compl, i_log_jump = [0,1,2]

inds_all = 'i_int, i_float, i_il, i_ol, i_res_buf = 0,1,2,3,4'
inds_int = 'i_ctr, i_ctrs, i_n_start, i_n_last, i_jump, i_glide, i_end_cond = [0,1,2,3,4,5,6]'
inds_float = 'i_slope, i_compl, i_log_jump = [0,1,2]'



keys_rec = ('mad', 'path_records', 'cycle_ns_min')
keys_range = ('mad', 'cycle_ns_min', 'cycle_ctrs', 'div_ctr', 'undef_ctr', 'stat_list', 'records', 't_calc_range')
keys_inf =  ('mad', 'slope_avg', 'compl_avg', 'log_dispersion', 'slope_std', 'op_types', 'op_freqs', 'op_freq_stds', 
                  'slope_bins', 'slope_hist', 'op_freqs_hist', 'ols', 'drops', 'drop_sk', 'drop_kur', 'len_block', 'n_iter_tot', 't_calc_slope', 'ec_wrong')


def mad_to_str(mad):
 '''
 Convert (mult, add, divisors) to string (used in automated results saving
 '''	
 mult, add, divisors = mad
 div_str = '-'.join([str(x) for x in divisors])
 s = f"{mult}_{add}_{div_str}"
 return s

def val_check(data_dict, ks, mad=None):
 '''
 Check if database entry has all values returned by avg_slope_calc, records_path_calc or stats_range
 Used in stats_calc function to determine if corresponding calculation is needed 
 '''
 ch = all([x in data_dict.keys() for x in ks])
 if not mad is None:
  ch = ch and data_dict['mad'] == mad
 return ch

def div_arr(arr_0, divisors=(2,)):
 '''
 Takes integer array arr_0 and divides it by divisos div until all numbers are not divisible by divs any more
 '''
 if isinstance(divisors, int):
  divisors = [divisors,]
 n_divs, n_nums = len(divisors), arr_0.size
 arr, div_ctrs = np.copy(arr_0), np.zeros((n_divs, n_nums), dtype=int)
 for i_div in range(n_divs):
  div = divisors[i_div]
  cont = np.ones_like(arr_0).astype(bool)
  while np.any(cont):
   i_cond = np.nonzero(cont)[0]
   res, rem = np.divmod(arr[i_cond], div)
   mask_stop = rem.astype(bool) #numbers which divided
   mask_cont = True ^ mask_stop
   cont[i_cond[mask_stop]] = False
   arr[i_cond[mask_cont]] = res[mask_cont]
   div_ctrs[i_div, i_cond[mask_cont]] += 1
  ratios = arr_0 / arr
 return arr, ratios, div_ctrs


def erato(n): #5M / 1s, 50M/15s #(+)
 '''
 Returns prime numbers up to n
 '''
 all_nums = np.arange(n+1)
 prime_mask = np.ones(n+1, dtype=np.bool_)
 for i_n in range(2, n):
  if not prime_mask[i_n]:
   continue
  inds_mult = i_n * all_nums[2:int(n/i_n)+1]
  prime_mask[inds_mult] = False 
 return [int(x) for x in all_nums[prime_mask][2:]]

primes = erato(int(1e6))

def parabola_vertex_calc(xs, ys):
 '''
 Calculates position of extremum of parabola defined by three xs and ys values, on x-axis
 '''
 par = np.polyfit(xs, ys, 2)
 k, b = 2*par[0], par[1] 
 v = -b/k
 return v
 

def div_prob_calc(divs):
 '''
 Calculates divisibility probabilities of a random number by list/tuple of prime divisors divs
 Returns np array probs[0] where probs[1:] are probabilities for each divisor and probs[0] = probability that a number is not divisible by any of them
 '''
 if len(divs) == 0:
  return np.array([1.0,])
 ds = np.array(divs)
 invs = 1 / ds
 negs = 1 - invs
 probs = np.zeros(len(divs) + 1)
 probs[1] = 1/2
 for i in range(1, probs.size - 1):
  probs[i+1] = negs[:i].prod() * invs[i]
 probs[0] = 1 - probs[1:].sum()
 return probs


def test_arr(n_test=1000, log_r_min=-2, log_r_max=3):
 r_min, r_max = 10**log_r_min, 10**log_r_max
 rs, phis = 10**np.random.uniform(-2, 3, n_test), np.random.uniform(0, 2*pi, n_test)
 phis[np.random.randint(0, n_test, int(n_test/4))] = 0
 phis[np.random.randint(0, n_test, int(n_test/4))] = pi
 zs = rs * np.exp(1j*phis)
 i_int = np.random.randint(0, n_test, int(n_test/4))
 zs[i_int] = np.random.randint(-int(r_max), int(r_max), int(n_test/4))
 return zs


def modal_calc(vs, n_bins=1000, n_sm=10, side='mid'):
 h, b = np.histogram(vs, bins = n_bins)
 h_sm = np.convolve(h, np.ones(n_sm)/n_sm, 'same')
 i_max = h_sm.argmax()
 inds_max = np.arange(i_max-n_sm, i_max+n_sm+1) if side == 'mid' else (np.arange(i_max, i_max+2*n_sm+1) if side == 'right' else np.arange(i_max - 2*n_sm, i_max+1))
 modal = Poly(np.polyfit(b[inds_max] + 0.5*(b[1] - b[0]), h[inds_max], deg=2)[::-1]).deriv().roots()
 return modal


def gamma_inv(y):
 if isinstance(y, np.ndarray):
  rs = np.zeros_like(y)
  for i in range(y.size):
   rs[i] = gamma_inv(y[i]) if y[i] >= 2 else 1.0
  return rs
 func = lambda x: scp.special.gamma(x) - y
 return scp.optimize.bisect(func, 2, 50)

def sem_calc(vs, n_rep=100, sub_len_rel=0.5, sub_len=100500):
 '''
 bootstrap calculation of standard error of the mean
 '''
 means = []
 mean = np.mean(vs)
 n_vals = len(vs)
 sl = int(n_vals * sub_len_rel)
 if 0 < sub_len_rel < 1 and sub_len > 1:
  subsample_len = min(sub_len, sl)
 elif sub_len > 1:
  subsample_len = sub_len
 elif 0 < sub_len_rel < 1:
  subsample_len = sl
 else:
  subsample_len = int(n_vals/2) 
 
 bs_ratio = subsample_len / n_vals 
 subsamples = np.random.choice(vs, subsample_len * n_rep).reshape((subsample_len, n_rep))
 means = np.mean(subsamples, axis=0)
 sem_raw = np.std(means, ddof=1)
 sem = sem_raw * sqrt(bs_ratio) 
 return mean, sem


def range_gen(log_range = (10.0, 15.0), n_sample=9000, divisors=[2,]):
 log_n_min, log_n_max = min(log_range), max(log_range)
 #if log_n_min < 0: call recursive
  
 odd_only = len(divisors) > 0
 n_min, n_max = int(10**log_n_min), int(10**log_n_max)
 p_odd = div_prob_calc(divisors)[0]
 
 n_full = n_max - n_min
 full_range = log_n_min < 20 and n_sample > n_full * p_odd
 use_mp = log_n_max > 15
 
 if not full_range:
  if use_mp:
   dps_0 = mp.dps
   mp.dps = round(log_n_max) + 10
   f_rand = mp_random
  else:
   f_rand = random.random
  ns = []
  ctr = 0
  while ctr < n_sample:
   log_rel = f_rand()
   log_n = log_n_min + log_rel * (log_n_max - log_n_min)
   n = int(10**log_n)
   if odd_only and any([n % x == 0 for x in divisors]):
    continue
   ns.append(n)
   ctr += 1
  ns = sorted(list(set(ns)))
  if use_mp:
   mp.dps = dps_0
 else:
  ns = [x for x in range(n_min, n_max) if not any([x % d == 0 for d in divisors])]

 return ns


def cycle_rotate(ccl_raw):
 len_ccl = len(ccl_raw)
 n_min = min(ccl_raw)
 i_min = ccl_raw.index(n_min)
 if i_min == 0:
  return [x for x in ccl_raw[:-1]]
 ccl = ccl_raw[i_min:len_ccl-1] + ccl_raw[:i_min]
 return ccl

def cycle_freq_est(ccl):
 fr = 0.0
 for i in range(len(ccl)):
  ccl_i = ccl[i]
  if abs(log10(ccl_i)) > -30:
   fr += 1 / ccl_i
 return fr


#taken from math_custom
def signum(x): #(+)
 return 1 if x > 0 else (-1 if x < 0 else 0) 

def hist_with_means(vals, n_bins=100, n_tails=20, nd_round=8, ftype=''):
 vs_round = np.round(vals, nd_round) if nd_round > 0 else vals
 vals_u, counts = np.unique(vs_round, return_counts=True)
 n_vals = vals_u.size
 
 if n_vals < n_bins + 2*n_tails:
  return (vals_u, counts/counts.sum()) if 'rel' in ftype else (vals_u, counts)

 bins = np.linspace(vals_u[n_tails], vals_u[n_vals-n_tails-1], n_bins)  # OK
 cts = np.zeros(n_bins + 2*n_tails, dtype=np.int64)
 means = np.zeros_like(cts).astype(float)
 for i in range(n_bins-1):
  vals_mask = np.where((bins[i] < vals_u) * (vals_u < bins[i+1]), True, False)
  cts[n_tails + i] = counts[vals_mask].sum()
  means[n_tails + i] = (counts[vals_mask] / np.sum(counts[vals_mask]) * vals_u[vals_mask]).sum()

 if n_tails > 0:
  cts[:n_tails] = counts[:n_tails]
  cts[-n_tails:] = counts[-n_tails:]
  means[:n_tails] = vals_u[:n_tails]
  means[-n_tails:] = vals_u[-n_tails:]

 if 'rel' in ftype:
  cts = cts / cts.sum()
 
 i_nz = np.nonzero(cts)[0]
  
 return means[i_nz], cts[i_nz]

'''
 vals_u, counts = np.unique(np.round(vals, nd_round), return_counts=True)
 n_vals = vals_u.size
 bins = np.linspace(vals_u[n_tails], vals_u[-n_tails], n_bins)  # OK
 cts = np.zeros(n_bins + 2*n_tails, dtype=np.int64)
 vals_mid = vals[n_tails:n_vals-n_tails]
 bins = np.linspace(np.min(vals_mid), np.max(vals_mid), n_bins)  # OK
 bin_indices = np.digitize(vals_mid, bins) - 1 #OK
 counts = np.bincount(bin_indices, minlength=len(bins)-1)
 sums = np.bincount(bin_indices, weights=vals_mid, minlength=len(bins)-1)
 means = np.where(counts > 0, sums / counts, np.nan)

 counts_all = np.hstack(( np.ones(n_tails).astype(int), counts, np.ones(n_tails).astype(int) ))
 means_all = np.hstack(( vals[:n_tails], means, vals[n_vals-n_tails:] ))

 i_nz = np.nonzero(counts_all)[0]
 ms = means_all[i_nz]
 cts = counts_all[i_nz] if 'rel' not in ftype else (counts_all[i_nz] / n_vals)
'''

def merge_hists(vals1, cts1, vals2, cts2, n_bins=-1, n_tails=30, nd_round=4, ftype=''):
 if n_bins > 0 and nd_round > 0: #re-create histogram with new parameters, use with care
  return merge_hists_brute(vals1, cts1, vals2, cts2, n_bins, n_tails, nd_round, ftype)
 vals_all = np.union1d(vals1, vals2)
 cts_all = np.zeros_like(vals_all, dtype=np.int64)
 inds1 = np.searchsorted(vals_all, vals1)
 np.add.at(cts_all, inds1, cts1)
 inds2 = np.searchsorted(vals_all, vals2)
 np.add.at(cts_all, inds2, cts2)
 return vals_all, cts_all

def merge_hists_brute(vals1, cts1, vals2, cts2, n_bins=-1, n_tails=-1, nd_round=8, ftype='min'):
 if n_bins <= 0 or n_tails <= 0:
  n_all_1 = vals1.size
  n_tails_1 = int(np.where(cts1 == 1)[0].size/2)
  n_bins_1 = n_all_1 - n_tails_1
  n_all_2 = vals2.size
  n_tails_2 = int(np.where(cts2 == 1)[0].size/2)
  n_bins_2 = n_all_2 - n_tails_2
  n_bins = max(n_bins_1, n_bins_2) if 'max' in ftype else min(n_bins_1, n_bins_2)
  n_tails = max(n_tails_1, n_tails_2) if 'max' in ftype else min(n_tails_1, n_tails_2)

 vals_combined = np.concatenate((np.repeat(vals1, cts1), np.repeat(vals2, cts2)))
 vals, cts = hist_with_means(vals_combined, n_bins=n_bins, n_tails=n_tails, nd_round=nd_round, ftype=ftype)
 return vals, cts

def bin_search(func, y_val, x_range=(0.0,1.0), prec=1e-6, ftype='', deg=1, range_expand=1.5, max_ext=1e20, ctr_lim=100): 
 x_l, x_r = x_range
 
 f_types = ftype.split(' ')
 delta_x = diff_start = abs(x_r - x_l) 

 y_l, y_r = func(x_l), func(x_r)
 yd_l, yd_r = y_l - y_val, y_r - y_val
 xs, ys = [x_l, x_r], [y_l, y_r]
 cond = True
 
 #---------------range extension if needed and indicated by ftype 
 
 if signum(yd_l) == signum(yd_r):
  if 'ext' not in f_types:
   x_val = nan
   cond = False
  else:
   diff_rel = 1.0 #relative difference, 1.0 @ the start
   while signum(yd_l) == signum(yd_r) and abs(delta_x / diff_start) < max_ext and any((yd_l != 0, yd_r != 0)): #x_l if abs(yd_l) < abs(yd_r) else x_r
    delta_x = range_expand * delta_x
    if abs(yd_l) < abs(yd_r): #move to the left
     x_l, x_r = x_l - delta_x ,x_l
     y_l, y_r = func(x_l), y_l
     xs.append(x_l)
     ys.append(y_l)
    else: # abs(yd_l) > abs(yd_r): #move to the right
     x_l, x_r = x_r, x_r + delta_x
     y_l, y_r = y_r, func(x_r)
     xs.append(x_r)
     ys.append(y_r)
    yd_l, yd_r = y_l - y_val, y_r - y_val
 
 #---------------main search ----------
 
 ctr=0
 while cond:
  x_m = 0.5*(x_l+x_r)
  y_m = func(x_m)
  xs.append(x_m)
  ys.append(y_m)
  yd_m = y_m - y_val
  if signum(yd_l) == signum(yd_m): # (pt2 -> pt1)
   x_l, y_l, yd_l = x_m, y_m, yd_m
  else: #   signum(yd_m) == signum(yd_r):
   x_r, y_r, yd_r = x_m, y_m, yd_m 
  ctr += 1
 
  #---break conditions
 
  diff_rel = abs((x_r - x_l)/diff_start)#  if 'print' in ftype:#   print(ctr, '%.6e' % x_l, '%.6e' % x_m, '%.6e' % x_r, '%.2e' % diff, '\t', '%.6e' % y_l, '%.6e' % y_m, '%.6e' % y_r) #  print(ctr, x_l, x_m, x_r, y_l, y_m, y_r)  
  if any((yd_l == 0, yd_r == 0, yd_m == 0)):
   x_val = x_l if yd_l == 0 else (x_m if yd_m == 0 else x_r)
   cond = False
  if diff_rel < prec:
   break
  if ctr > ctr_lim or signum(yd_l) == signum(yd_r) and yd_l != 0:
   x_val = nan
   cond = False
 #-------------final calculation-------------- 

 if cond:
  x_3, x_2, x_1 = xs[-3:]
  y_3, y_2, y_1 = ys[-3:]
  yd_3, yd_2, yd_1 = y_3 - y_val, y_2 - y_val, y_1 - y_val
  if deg == 0:
   x_val = x_1
  elif deg == 1:
   x_val = x_1 - yd_1 * (x_1 - x_2) / (y_1 - y_2)
  else:
   xs_fin, yds_fin = np.array((x_3, x_2, x_1)), np.array((yd_3, yd_2, yd_1))
   sqr_last = sqr_calc(xs_fin, yds_fin)
   r_1, r_2 = sqr_solve(sqr_last, 0)
   x_val = r_1 if abs(r_1 - x_m) < abs(r_2 - x_m) else r_2

 if 'log' in f_types:
  xs_out = np.array(xs)
  ys_out = np.array(ys)
  return x_val, np.array((xs_out, ys_out)).T
 else:
  return x_val


def db_keys(db, ftype='', cond='', adds=(1,)):
 '''
 Gives list of keys of database db which are valid entries (tuples of (mult, add, (divisors)) ) and correspond to condition cond
 '''
 adds_all = adds == 'all'
 ks_all = list(db.keys())
 ks_valid = [x for x in ks_all if isinstance(x, tuple) and len(x) == 3 and isinstance (x[2], tuple)]
 if not adds_all:
  cond += (' and ' if len(cond) > 0 else '') + 'db[x][\'mad\'][1] in adds'
 if len(cond) > 0:
  ks_valid = [x for x in ks_valid if eval(cond)]
 ks_valid.sort() #remaining: implement sorting as in mad_combos
 return ks_valid if 'tuple' not in ftype else tuple(ks_valid)

def db_load(fn, cond='', ftype=''):
 with open(fn, 'rb') as d:
  db = pickle.load(d)
 if len(cond) > 0:
  ks_pass = db_keys(db, ftype, cond)
  db = {k:db[k] for k in ks_pass} 
 return db

def db_save(db, fn):
 with open(fn, 'wb') as d:
  pickle.dump(db, d)
 return None

def db_del(vs, ftype=''):

 if isinstance(ftype, tuple):
  for i in range(len(ftype)):
   db_del(vs, ftype[i])

 ks_del = keys_inf if ftype == 'inf' else keys_rec if ftype == 'rec' else keys_range 
 for k in ks_del:
  if k in vs.keys() and k != 'mad':
   p = vs.pop(k)

 return None
 
def common_nums(*arrs):
 nums = arrs[0]
 for i in range(1, len(arrs)):
  nums = np.intersect1d(nums, arrs[i])
 return nums
 
#(+)
#taken from math_custom->mc_gen
#calculates rational approximation of a number, using continuous fraction
#additional parameters, order of influence: length -> prec -> max denominator -> cont frac threshold
def cont_frac(x_in, length=30, prec=0.0, ftype='ext', den_max=inf, best=False, cf_thr=0): #50kcalc/s / 10 iters
 x = x_in
 q = floor(x)
 x = x - q
 delta = 1
 ctr=0
 n, d, num, den = 0, 1, 1, 0
 nums, dens, cf = [], [], []
 while ctr < length and den <= den_max and delta > prec and 1 / x != q:
  ctr += 1
  nums.append(num)
  dens.append(den)
  cf.append(q)
  n, d, num, den = num, den, num*q + n, den*q + d
  q = floor(1 / x)
  x = 1 / x - q
  delta = abs(nums[-1] / (dens[-1]*x_in) - 1) if ctr > 1 else 1 #(abs(rationaltoreal(cftorational(cf))/x_in - 1) if prec > 0 else 1) 
  
 if ftype=='cf':
  return np.array(cf)
 
 nums, dens, cf = np.array(nums), np.array(dens), np.array(cf)# i_0 = 2 if (nums[1], dens[1]) == (0,1) else 1
 i_cf_max = 1 + np.argmax(cf[1:])
 cf_max = cf[i_cf_max]
 if best: #choose best continued fraction if maximum above threshold
  i_fr = i_cf_max if cf_max >= cf_thr else -1
 else: #choose last cf above threshold or last value
  i_fr = -1 if cf_thr == 0 or cf_max < cf_thr else 1 + np.where(cf[1:] >= cf_thr)[0][-1] 

 num, den, cf_next = nums[i_fr], dens[i_fr], cf[i_fr] #  cf_next = (cf[i_fr+1] if i_fr < cf.size-1 else cf_next)

 if ftype=='seq':
  return np.vstack((nums, dens))
 elif ftype=='all':
  return np.vstack((nums, dens, cf))
 elif ftype=='ext':
  return np.array((num, den, cf_next))
 else:
  return np.array((num, den))


class Records:

 def __init__(self, n_max=20):
  self.jumps = []
  self.compls = []
  self.slopes = []
  self.maxlen = n_max
  self.t_start = time.time()
  self.n_sample = 0

 def list_update(self, vals_cur, ftype='slope'):
  i_val = i_compl if 'compl' in ftype else (i_log_jump if 'jump' in ftype else i_slope)
  vs_list = self.compls if 'compl' in ftype else (self.jumps if 'jump' in ftype else self.slopes)
  i_st = 0 if len(vs_list) < self.maxlen else 1
  val = vals_cur[i_float][i_val]
  if val < vs_list[-1][i_float][i_val] and 'record' in ftype or (val < vs_list[0][i_float][i_val] and i_st == 1):
   return None
  vals = [x[i_float][i_val] for x in vs_list]
  i_ins = bisect.bisect_left(vals, val)
  vs_list = vs_list[i_st:i_ins] + [vals_cur[:2],] + vs_list[i_ins:]
  if 'compl' in ftype:
   self.compls = vs_list
  elif 'jump' in ftype:
   self.jumps = vs_list
  else:
   self.slopes = vs_list

  if 'print' in ftype:
   print(f"{val:.8f} {ftype}, at {time.time - self.t_start:.2f} seconds, n_sample {self.n_sample}")
  
  return None
   
 def upd(self, vals_cur, ftype=''):
  record_type = ' record' if 'record' in ftype else ''
  self.n_sample += 1
  slope, compl, jump = vals_cur[i_float]
  
  if len(self.jumps) == 0 or len(self.compls) == 0 or len(self.slopes) == 0:
   self.jumps.append(vals_cur[:2])
   self.compls.append(vals_cur[:2])
   self.slopes.append(vals_cur[:2])
   return None
 
  self.list_update(vals_cur, ftype='slope' + record_type)
  self.list_update(vals_cur, ftype='compl' + record_type)
  self.list_update(vals_cur, ftype='jump' + record_type)
  
 def __repr__(self):
  if len(self.jumps) == 0 and len(self.slopes) == 0 and len(self.compls) == 0:
   return ''
  n_jumps, jumps = [x[i_int][i_n_start] for x in self.jumps][::-1], [x[i_float][i_log_jump] for x in self.jumps][::-1]
  n_slopes, slopes = [x[i_int][i_n_start] for x in self.slopes][::-1], [x[i_float][i_slope] for x in self.slopes][::-1]
  n_compls, compls = [x[i_int][i_n_start] for x in self.compls][::-1], [x[i_float][i_compl] for x in self.compls][::-1]
  str_out = 'Slope records:\n'
  for i in range(min(11, len(slopes))):
   str_out += f"{slopes[i]:.6g} {n_slopes[i]}\n"
  str_out += '\nJump records:\n'
  for i in range(min(11, len(jumps))):
   str_out += f"{jumps[i]:.6g} {n_jumps[i]}\n"
  str_out += '\nCompleteness records:\n'
  for i in range(min(11, len(compls))):
   str_out += f"{compls[i]:.6g} {n_compls[i]}\n"
  return str_out

 def max_vals(self):
  max_jump = self.jumps[-1][i_float][i_log_jump] if len(self.jumps) > 0 else nan
  max_compl = self.compls[-1][i_float][i_compl] if len(self.compls) > 0 else nan
  max_slope = self.slopes[-1][i_float][i_slope] if len(self.slopes) > 0 else nan

  max_jump_num = self.jumps[-1][i_int][i_n_start] if len(self.jumps) > 0 else nan
  max_compl_num = self.compls[-1][i_int][i_n_start] if len(self.compls) > 0 else nan
  max_slope_num = self.slopes[-1][i_int][i_n_start] if len(self.slopes) > 0 else nan
  return ((max_slope, max_compl, max_jump), (max_slope_num, max_compl_num, max_jump_num))
  
 def arr(self):
  return np.array(self.slopes), np.array(self.jumps), np.array(self.compls)



'''

  self.min_jump = 0.0
  self.max_jump = 0.0
  self.len_jump = 0
  self.min_slope = 0.0
  self.max_slope = 0.0
  self.len_slope = 0
  self.min_compl = 0.0
  self.max_compl = 0.0
  self.len_compl = 0 

'''
