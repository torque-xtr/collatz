from collatz import *

def bin_viz(arr, ftype='clf'):
 if 'clf' in ftype:
  plt.clf()
 n_fl = arr.astype(float)
 n_smooth = np.convolve(n_fl, np.ones(5)/5, mode='same')
 plt.plot(np.arange(n_fl.size), n_smooth)
 plt.scatter(np.arange(n_fl.size), n_fl, s=2)
 return None


def log_viz_bool(v, asp_ratio=0.25, ftype='', wrap_len=1.0, roll=1):
 n = to_arr(v) if not isinstance(v, np.ndarray) else v
 vs = collatz_bin(v, ftype='log')

 log_iter = vs[0]
 log_lens = [x.size for x in log_iter]
 max_len, len_log = max(log_lens), len(log_iter)
 log_width = max_len + len_log if 'full' in ftype else max_len
 add_width = max(0, int((wrap_len - 1) * max_len)) if 'wrap' in ftype else 0
 log_arr = np.zeros((len_log, log_width + add_width), dtype=np.bool_)
 
 i_start=0
 for i in range(len_log):
  log_arr[i, i_start: i_start + log_iter[i].size] = log_iter[i][::-1]   
  if 'full' in ftype:
   i_start += 1
  if 'wrap' in ftype:
   log_arr[i] = np.roll(log_arr[i], i*roll)
 
 bool_viz(True ^ log_arr[::-1, ::-1].T)
 
 plt.gca().set_aspect(asp_ratio)
 
 return vs[1:]


def log_viz_2D(n, mad=(), mult=3, add=1, divisors=[2,], ctr_lim=int(1e7), log_n_max=1000,
               roll=0, base=-1, n_digits=-1, ftype=''):
  
 if len(mad) == 3:
  mult, add, divisors = mad
  
 base = base if base > 1 else prod(divisors)
 il, ol, ec = collatz(n, mult=mult, divisors=divisors, add=add, log_n_max=log_n_max, ctr_lim=ctr_lim) #OK, debugged

 if 'odd' in ftype:
  il = [il[i] for i in range(len(il)) if ol[i] == 1]
 
 len_log = len(il)
 n_max = max(il)
 n_digits_0 = ceil(log(n_max, base) * 2)
 n_digits = max(n_digits_0, n_digits)
 arr_viz = np.zeros((len_log, n_digits), dtype=int if base==2 else float)

 i_start=0
 for i in range(len_log):
  n_cur = il[i]
  n_arr = to_arr_large(n_cur, base=base)
  arr_viz[i,n_digits-n_arr.size:] = n_arr


 arr_norm = arr_viz / base if base != 2 else arr_viz
 if roll != 0:
  for i in range(len_log):
   arr_norm[i] = np.roll(arr_norm[i], i*-roll)
 
 return il, arr_norm


'''
 if 'wrap' in ftype: #debug if needed
  add_width = max(0, int((wrap_len - 1) * max_len))
  log_arr_2 = np.zeros((len_log, log_width + add_width), dtype=np.bool_)
  for i in range(len_log):
   line_cur = np.zeros(log_width + add_width, dtype=np.bool_)
   line_cur[:log_width] = log_arr[i]
   log_arr_2[i] = np.roll(line_cur, i*roll)
  
  bool_viz(True ^ log_arr_2[::-1, ::-1].T)
    
 else:

'''


def bool_viz(vs, xs=None, ys=None, ftype='clf', plt_name='', clr_true='b', clr_false='r', thr=1.0):
 len_x, len_y = vs.shape
 if type(xs) != np.ndarray:
  xs_v, ys_v = np.indices((len_x, len_y))
 elif xs.ndim == 1:
  xs_v, ys_v = np.meshgrid(xs, ys, indexing='ij')
 else:
  xs_v, ys_v = xs, ys
 vs_v = vs if vs.dtype == bool else np.where(vs_v > thr, True, False)
 plt.ion()
 plt.show()
 if 'clf' in ftype:
  plt.clf()
 
 clrs = np.zeros((len_x, len_y, 4))
 clrs[vs] = mpl.colors.to_rgba(clr_true)
 clrs[True ^ vs] = mpl.colors.to_rgba(clr_false)
 p = plt.imshow(np.transpose(clrs, (1,0,2)), aspect='auto', interpolation='none', origin='lower', extent=[xs_v[0,0],xs_v[-1,0],ys_v[0,0],ys_v[-1,-1]]) 

 if plt_name != '':
  plt.savefig(plt_name, dpi=200)
 return None
