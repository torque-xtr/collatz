from collatz_aux import *
from scipy.stats import binomtest
from scipy.stats import normaltest
from scipy.stats import norm
from scipy.optimize import root_scalar
from scipy.optimize import minimize_scalar
from scipy.interpolate import CubicSpline
from scipy.misc import derivative


'''
Auxiliary functions used in calculations of maximum slope and jump for generalized Collatz sequences, based on the assumption that iteration sequences are Markov chain.
Slope = (log10(n_end) - log10(n_start)) / number of iterations
Jump = log10(sequence maximum) / log10(n_start)
For (3,1,(2,)) Collatz sequence, maximum jump is empirically observed to be very close to 2.
A 'raw_drops'-type calculation with limits_calc function shows that it converges to 2 as log(n_start) increases
'''

def ol_sim(len_log):
 '''
 Used in first-type slope and jump limits calculations
 Simulate operation type log of classic Collatz sequence, in term of lengths of descending (division-type) sequences
 '''
 ps = np.random.uniform(0, 1, len_log)
 ns = np.log2(ps) * (-1)
 ns_int = np.ceil(ns).astype(int)
 return ns_int

def desc_drop_calc(ol, ftype=''):
 '''
 Calculates drop values for descending subsequences of operation type log ol 
 Returns list of all drops corresponding to odd log, or mean drop if 'mean' in ftype
 'drop' = log10(n_start) - log10(n_end) 
 Descending sequence contains only divide-type operations, e,g, [10,5] and [16,8,4,2,1] for [3,10,5,16,8,4,2,1]  sequence
 '''
 cond = True
 log_drops = []
 ol_arr = np.array(ol) if not isinstance(ol, np.ndarray) else ol
 is_odd = np.where(ol_arr == 0)[0]
 i_last = is_odd[-1]
 for i in range(is_odd.size - 1):
  i_l = is_odd[i] + 1 
  i_r = is_odd[i+1] 
  log_drop = -1.0*log10(prod(ol[i_l:i_r])) #log_drop = log10(il[i_r]) - log10(il[i_l])
  log_drops.append(log_drop)
 return np.mean(log_drops) if ftype=='mean' else log_drops


#part of pdf_emp
def rate_func(val, vals_u, val_freqs=None):
 '''
 Calculates rate function in prob(val > vals) = exp(-N*rane_func(val))
 Takes: threshold value, theoretical/observed values vals_u(raw values, unique values, or histogram bin averages, and their counts/frequencies (normalized in cgf_calc)
 Returns: max of Legengre-Fenchel transform (lft function), value of rate function at 'val'
 '''
 if val < vals_u.min():
  theta_max = -inf
 elif val > vals_u.max():
  theta_max = inf
 else:
  theta_max = lft_max(val, vals_u, val_freqs)
 rate_val = lft(val, theta_max, vals_u, val_freqs) if isfinite(theta_max) else inf
 return theta_max, rate_val


def pdf_calc(ctr_eff, drop, drops_u=None, drop_freqs=None, ftype='log'):
 '''
 Calculates probability that a sequence of ctr_eff elements would have mean drop drop or higher if sistribution of drops is set by unuque values drops_u and their frequenciec drop_freqs
 Can be used with single iteration-type or block-type calculation
 '''
 theta_max, rate_val = rate_func(drop, drops_u, drop_freqs)
 if rate_val == -inf:
  return -inf if 'log' in ftype else 0.0
 sigma_2m, probs_t = tilted_var(theta_max, drops_u, drop_freqs) #
 corr = 1 / sqrt(2 * pi * ctr_eff * sigma_2m)
 pr_log = -ctr_eff * rate_val + log(corr)
 return pr_log * log10(e) if 'log' in ftype else exp(pr_log)

def prob_calc_gauss(log_n_start, log_n_end, ctr=1e5, len_block=1000.0, slope_avg=-0.07, log_dispersion=0.3, drop_sk=0.0, drop_kur=0.0, ftype='corr log'):
 drop_avg, drop_std = len_block * slope_avg, sqrt(len_block) * log_dispersion
 delta_log = log_n_end - log_n_start
 n_blocks = ctr / len_block
 drop_thr = (log_n_end - log_n_start) / n_blocks 
 std_tot = drop_std / sqrt(n_blocks)
 n_sigma = (drop_thr - drop_avg) / std_tot 
 log_prob = cdf_exp(-n_sigma) #reverse to simplify calculations
 if 'corr' in ftype and (drop_sk != 0.0 or drop_kur != 0.0): #wrong, fix it later
  log_pdf_val = pdf_exp(n_sigma)
  sk_term  = (n_sigma**2 - 1) * drop_sk / (6*sqrt(n_blocks))
  kur_term = (n_sigma**4 - 6*n_sigma**2 + 3) * drop_kur / (24*n_blocks)
  corr = (1 + sk_term + kur_term)
  log_corr = log10(corr) if corr > 0 else -inf  
  log_prob += log_corr
 return log_prob if 'log' in ftype else 10**log_prob
 
def prob_calc(log_n_start, log_n_end, ctr, drops_u=None, drop_freqs=None, ftype='log'):
 '''
 The same as pdf_calc but if sequence length and log10 of starting and ending numbers of a Collatz sequence are specified
 ctr is effective sequence length: number of blocks or 'odd + even' + 'pure even' iterations
 In the latter case, ctr_eff is calculated based on expected completeness (odd/even ratio; can be complicated)
 '''
 if ctr < 0:
  return -inf if ftype == 'log' else 0.0
 slope_eff = (log_n_end - log_n_start) / ctr
 if slope_eff > drops_u.max() or slope_eff < drops_u.min():
  return -inf if ftype == 'log' else 0.0
 prob = pdf_calc(ctr, slope_eff, drops_u=drops_u, drop_freqs=drop_freqs, ftype=ftype)
 return prob
  
#DS  
def tilted_var_raw(theta, drops):
 '''
 Calculate tilted variance for observed drops and a value of theta
 '''
 weights = np.exp(theta * drops)
 mean = np.sum(drops * weights) / np.sum(weights)
 variance = np.sum((drops - mean)**2 * weights) / np.sum(weights)
 return variance
 
def tilted_var(theta, drops_u, drop_freqs, ftype=''):
 '''
 Calculate tilted variance for observed unique drop values and their frequencies (or a normalized histogram) and a value of theta
 '''
 exp_terms = np.exp(theta * drops_u)
 tilted_vals = exp_terms*drop_freqs
 tilted_probs = tilted_vals / tilted_vals.sum()
 mean = (drops_u*tilted_probs).sum()
 mean_sq = (drops_u**2 * tilted_probs).sum()
 variance = mean_sq - mean**2
 return variance, tilted_probs
  
 #see https://chat.deepseek.com/a/chat/s/dc8f2828-6f1b-4aae-9c3e-f65e82e5cc92 
def tilted_prob_calc(drop, drops_u, drop_freqs):
 '''
 Calculate theoretical tilted drop frequencies for a drop value
 Used in drop->completeness calculation by compl_calc function;
 Deviates from observed frequencies for generalized-type Collatz sequences
 '''
 f_opt = lambda th: (cgf_d1_calc(th, drops_u, drop_freqs) - drop)**2
 theta = scp.optimize.minimize_scalar(f_opt, bounds=(0, 10), method='bounded').x
 probs = np.zeros_like(drop_freqs)
 for i_d in range(drops_u.size):
  numer = np.exp(theta*drops_u[i_d]) * drop_freqs[i_d] 
  denom = ( np.exp(theta*drops_u) * drop_freqs).sum()
  probs[i_d] = numer/denom
 return probs

#------------------empirical estimations----------------

def cdf_exp(sigma, ftype='log'):
 val_exp = log(1/sqrt(2*pi)) - log(-sigma) - sigma**2/2 # (1/√(2π)) * (1/x) * exp(-x²/2)  
 val_10 = val_exp * log10(e)
 return val_10 if 'log' in ftype else 10**val_10

def pdf_exp(sigma, ftype='log'):
 log_val = (-sigma**2/2 - 0.5*log(2*pi))*log10(e)
 return log_val if 'log' in ftype else 10**log_val
 
#calculate on raw sample of drops 
#check with -1.0 * drops / log10(2) from 3n+1/2-type odd log stats, and lt_check(theta)
def cgf_calc_raw(val, vals):
 '''
 Calculate cumulant generating function for a value theta and observed values 'drops' 
 '''
 exps = e**(val*vals)
 exp_sum = np.sum(exps)
 cgf = np.log(exp_sum/vals.size)
 return cgf
 
def cgf_calc(val, vals_u, val_freqs=None):
 '''
 Calculate cumulant generating function for a value val and normalized histogram of vals 'vals_u' and their weights/frequencies val_freqs
 '''
 if not ((isinstance(val_freqs, np.ndarray) and val_freqs.size == vals_u.size)):
  val_freqs = np.ones_like(vals_u) / vals_u.size
 val_freqs = val_freqs / val_freqs.sum()
 cgf = np.log(np.sum(val_freqs*np.exp(val*vals_u)))
 return cgf

def cgfs_calc(val, vals_u, val_freqs=None):
 '''
 Calculate cumulant generating function, it's 1st and 2nd derivatives for a value val and normalized histogram of vals 'vals_u' and their weights/frequencies val_freqs
 ''' 

 if not (isinstance(val_freqs, np.ndarray) and val_freqs.size == vals_u.size):
  val_freqs = np.ones_like(vals_u) / vals_u.size
 w_exps = val_freqs*np.exp(val*vals_u)
 val_0 = np.sum(w_exps)
 val_1 = np.sum(vals_u*w_exps)
 val_2 = np.sum(vals_u**2*w_exps)
 cgf = np.log(val_0)
 d1_cgf = val_1 / val_0
 d2_cgf = val_2 / val_0 - d1_cgf**2
 return cgf, d1_cgf, d2_cgf


def cgf_d1_calc(val, vals_u, val_freqs=None, w_exps=None):
 '''
 Calculate 1st derivative of cumulant generating function for a value val and normalized histogram of vals 'vals_u' and their weights/frequencies val_freqs
 '''
 if not (isinstance(val_freqs, np.ndarray) and val_freqs.size == vals_u.size): #no weights specified; raw values assumed
  val_freqs = np.ones_like(vals_u) / vals_u.size
 if not isinstance(w_exps, np.ndarray):
  w_exps = val_freqs*np.exp(val*vals_u)
 d1_cgf = np.sum(vals_u*w_exps) / np.sum(w_exps) 
 return d1_cgf


def cgf_d2_calc(val, vals_u, val_freqs=None, w_exps=None):
 '''
 Calculate 2nd derivative of cumulant generating function for a value val and normalized histogram of vals 'vals_u' and their weights/frequencies val_freqs
 '''
 if not ((isinstance(val_freqs, np.ndarray) and val_freqs.size == vals_u.size)):
  val_freqs = np.ones_like(vals_u) / vals_u.size
 if not isinstance(w_exps, np.ndarray):
  w_exps = val_freqs*np.exp(val*vals_u)
 d1_cgf = np.sum(vals_u*w_exps) / np.sum(w_exps) 
 d2_cgf = np.sum(vals_u**2*w_exps) / np.sum(w_exps) - d1_cgf**2
 return d2_cgf


def lt_check(theta):
 '''
 check for cgf calculation, old-type limits_calc (based on descending subsequence statistics)
 '''
 return theta - log(2 - e**theta)

def lft(val, theta, vals_u, val_freqs=None):
 '''
 Legendre-Fenchel transform for threshold value theta, value val, unique values/bin centers vals_u and frequencies/counts val_freqs (normalized in cgf_calc if needed)
 '''
 l = cgf_calc(theta, vals_u, val_freqs)
 v = theta*val - l
 return v


def lft_raw(val, theta, vals):
 '''
 The same as lft (see docstring) but with raw sample pf observed values 'vals'
 '''
 l = cgf_calc_raw(theta, vals)
 v = theta*val - l
 return v


def lft_max(val, vals_u, val_freqs=None, thr=1e-4):
 '''
 Iterative calculation of Legendre-Fenchel transform value val, unique values/bin centers vals_u and frequencies/counts val_freqs (normalized in cgf_calc if needed)
 Iterate maximum search until difference between last iterations < thr
 '''

 ths = [0.0, -0.5, 0.5]
 ls = [0.0,]
 for th in ths[1:]:
  ls.append(float(lft(val, th, vals_u, val_freqs)))

 d_th, ctr = 0.5, 0
 while d_th > thr and ctr < 20:
  ctr += 1
  parabola = np.polyfit(ths[-3:], ls[-3:], 2)
  a, b, c = parabola
  th = -b / (2*a)
  ths.append(float(th))
  ls.append(float(lft(val, th, vals_u, val_freqs)))
  d_th = max(abs(ths[-1] - ths[-2]), abs(ths[-1] - ths[-3]))
 
 return ths[-1]



#-------------------block-based estimation--------------


#---------------------------

def cdf_calc(vals, nd_round=8, bounds=(-inf, inf)):
 '''
 Calculate cumulative density function for observed sample of values vs
 '''
 vals_u, counts = np.unique(np.round(vals, nd_round), return_counts=True)
 cum_counts = np.cumsum(counts)
 n_vals = vals.size
 y_cdf = np.hstack((0.0, cum_counts/(n_vals+1), 1.0))
 x_cdf = np.hstack((bounds[0], vals_u, bounds[1]))
 return x_cdf, y_cdf
 


def prob_calc_block(*, log_n_start, log_n_end, ctr, len_block, drop_hist=None, drop_weights=None, drops=None, spls=None, ftype=''):
 '''
 Calculates probability that iteration log which starts with 10**(log_n_start) and ends with 10**(log_n_end), has more than ctr iterations (or slope greater than (log_n_end - log_n_start) / ctr, equivalently
 if a large sample of iteration log slices with len 'len_block' have histogram of drops (=log_n_end - log_n_start) specified by drop_hist and drop_weights
 Can take also raw values of obserfed drops as 'drop_hist'
 
 '''
 
 if ctr < 0:
  return -inf
  
 n_blocks = ctr / len_block
 drop_avg = (log_n_end - log_n_start) / n_blocks
 drop_max = drop_hist[-1] if isinstance(drop_hist, np.ndarray) else inf
 if drop_avg > drop_max:
  return -inf
 
 #calculation with splines (explicit calc fast enough usually -> not used)
 if isinstance(spls, tuple) and len(spls) == 3:
  spl_cgf, spl_cgf_d1, spl_cgf_d2 = spls
  saddle_pts = spl_cgf_d1.solve(drop_avg)
  s_IR = saddle_pts[np.where((saddle_pts < drop_max) * (saddle_pts > 0))]
  if s_IR.size == 0:
   if isinstance(drop_hist, np.ndarray):
    return prob_calc_block(log_n_start=log_n_start, log_n_end=log_n_end, ctr=ctr, len_block=len_block, drop_hist=drop_hist, drop_weights=drop_weights)
   else:
    return -inf
  saddle_pt = s_IR.max() if s_IR.size > 0 else nan
  cgf_val, cgf_d1_val, cgf_d2_val = float(spl_cgf(saddle_pt)), float(spl_cgf_d1(saddle_pt)), float(spl_cgf_d2(saddle_pt))
 
 else:
  f_saddle = lambda t: cgf_d1_calc(t, drop_hist, drop_weights) - drop_avg
  saddle_pt = scp.optimize.newton(f_saddle, x0=0.0)
  cgf_val, cgf_d1_val, cgf_d2_val = cgfs_calc(saddle_pt, drop_hist, drop_weights)
 if not isfinite(saddle_pt) or cgf_d2_val < 0:
  return -inf
 elif saddle_pt < 0:
  return 0.0
 else:
  log_prob = n_blocks * (cgf_val - drop_avg * saddle_pt) - 0.5 * log(2 * pi * n_blocks * cgf_d2_val) - log(saddle_pt)
  if 'corr' in ftype:
   K = lambda t: cgf_calc(t, drop_hist, drop_weights)
   s = scp.optimize.newton(lambda s: derivative(K, s) - drop_avg, x0=0.0)
   K2, K3, K4 = [derivative(K, s, n=n, dx=1e-2, order=n+(1 if n%2==0 else 2)) for n in [2, 3, 4]]
   term1 = K3 / (6 * np.sqrt(n_blocks) * K2**1.5)
   term2 = K4 / (8 * n_blocks * K2**2)
   term3 = 5 * K3**2 / (24 * n_blocks * K2**3)
   corr = 1 + term1 - term2 + term3
   if corr < 0:
    return -inf
   else:
    log_prob += corr
  return log_prob / log(10)


def prob_calc_is(*, log_n_start, log_n_end, ctr=2e5, drop_arr=None, ols=None, mad=None):
 '''
 The same as prob_calc and prob_calc_block but using importance sampling on observed odd logs 'ols', or array of drops 'drop_arr'
 Results agree with prob_calc_block but much slower.
 '''
 if isinstance(ols, list):
  ol_arr = np.array(ols) 
  drop_arr = np.where(ol_arr == 0, np.log10(mad[0]), np.log10(ol_arr)*-1.0)
 drops = drop_arr.sum(axis=1)
# i_srt = np.argsort(drops)
# ol_arr[i_srt[-20:]]

 drop_hist, dr_cts = hist_with_means(drops, n_bins=20, nd_round=3, ftype='')
 drop_weights = dr_cts/dr_cts.sum()
 #drops.max(), dr_max

 len_block = drop_arr.shape[-1]
 n_blocks = ctr / len_block
 drop_avg = (log_n_end - log_n_start) / n_blocks
 f_saddle = lambda t: cgf_d1_calc(t, drop_hist, drop_weights) - drop_avg
 saddle_pt = scp.optimize.newton(f_saddle, x0=0.0)

 tilt_wts_raw = np.exp(saddle_pt*drop_arr)
 tilt_wts = tilt_wts_raw / tilt_wts_raw.sum(axis=1)[:,None]

 drop_arr_tilted = drop_arr*tilt_wts
 drops_tilted = drop_arr_tilted.sum(axis=1)

 dr_frac = np.where(drops_tilted > drop_avg)[0].size / drops_tilted.size
 #dr_frac, drops_tilted.min(), drops_tilted.mean(), drops_tilted.max()

 cgf_val = cgf_calc(saddle_pt, drop_hist, drop_weights)
 prob = -n_blocks * (saddle_pt * drop_avg - cgf_val) + (log(dr_frac) if dr_frac > 0 else -inf)

 return prob * log10(e)
 
