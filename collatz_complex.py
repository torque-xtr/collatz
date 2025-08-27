from collatz_aux import * #all libraries are imported here


 
@njit
def collatz_iter_z(z, add=1, mult=3, div=2, odd_type=1):
 z_ = z*pi/2
 z_cos = z * cmath.cos(z_)**2 / div
 z_sin = (mult*z + add) * cmath.sin(z_)**2
 if odd_type==2: #divide by 2 to get wiki-style function
  z_sin /= 2
 z_mix =  (1/2 - cmath.cos(pi*z))*cmath.sin(pi*z)/ pi
 z_next = z_cos + z_sin + z_mix
 return z_next

#takes complex argument of mpmath type, very slow
def collatz_iter_mp(z, add=1, mult=3, div=2, odd_type=1):
 z_pi, z_pihalf = z*mp.pi, z*mp.pi/2
 z_cos = z * mp.cos(z_pihalf)**2 / div
 z_sin = (mult*z + add) * mp.sin(z_pihalf)**2 #divide by 2 to get wiki-style function
 if odd_type==2:
  z_sin /= 2
 z_mix =  (1/2 - mp.cos(z_pi))*mp.sin(z_pi)/ pi
 z_next = z_cos + z_sin + z_mix
 return z_next



#@njit
def collatz_z(z_0, add=1, mult=3, div=2, ctr_lim=1000, imag_max=1e2, real_max=1e7, cycle_thr=1e-6, zs_min = [1+0j,], odd_type=1):
 
 z, ctr, cond, z_log = z_0, 0, True, [z_0,]

 while cond:
  ctr += 1
  z = collatz_iter_z(z, add=add,  mult=mult, div=div, odd_type=odd_type) 
  z_log.append(z)

  if isnan(z.real) or isnan(z.imag):
   z_pop = z_log.pop(-1)
   break
  z_abs = abs(z)
  thr_cur = cycle_thr * max(abs(add), z_abs)
  if ctr > ctr_lim or abs(z.imag) > imag_max or abs(z.real) > real_max:   #divergence detection
   break
  if sum( [ (abs(z.real - x.real) < thr_cur and abs(z.imag - x.imag) < thr_cur) for x in zs_min] ) > 0:
   break
 
 return z_log



def collatz_mp(z_0, add=1, mult=3, div=2, ctr_lim=1000, imag_max=1e2, real_max=1e7, cycle_thr=1e-6, zs_min = [1+0j,], odd_type=1):
 
 z, ctr, cond, z_log = z_0, 0, True, [z_0,]

 while cond:
  ctr += 1
  z = collatz_iter_mp(z, add=add,  mult=mult, div=div, odd_type=odd_type) 
  z_log.append(z)

  if not (isfinite(z.real) and isfinite(z.imag)):
   z_pop = z_log.pop(-1)
   break
  z_abs = abs(z)
  thr_cur = cycle_thr * max(abs(add), z_abs)
  if ctr > ctr_lim or abs(z.imag) > imag_max or abs(z.real) > real_max:   #divergence detection
   break
  if sum( [ (abs(z.real - x.real) < thr_cur and abs(z.imag - x.imag) < thr_cur) for x in zs_min] ) > 0:
   break
 
 return z_log


#assumes z_log contains more than one z_min to select cycle correctly
def cycle_calc_z(z_log, cycle_thr=1e-6):

 len_log = len(z_log)
 rev_log = z_log[::-1]
 z_last = rev_log[0]
 
 thr = max(cycle_thr, abs(z_last)*cycle_thr)
 det = False
 for i_next in range(1, len_log):
  z_cur = rev_log[i_next]
  if isclose(z_cur.real, z_last.real, abs_tol=thr) and isclose(z_cur.imag, z_last.imag, abs_tol=thr):
   det = True
   break

 if det:
  cycle_raw = rev_log[:i_next][::-1]
  len_ccl = len(cycle_raw)
  zs_abs = [abs(z) for z in cycle_raw]
  i_min = zs_abs.index(min(zs_abs))
  z_min = cycle_raw[i_min]
  i_st = (i_min + 1) % len_ccl
  ccl = cycle_raw[i_st:] + cycle_raw[:i_st]
  return z_min, ccl
 else:
  return (nan+nan*1j), []

def cycle_find_z(zs, add=1, mult=3, div=2, ctr_lim=1000, imag_max=1e2, real_max=1e10, tol=1e-6, odd_type=1):
 zs_min, cycles = [], []
 for i_z in range(zs.size):
  z = zs[i_z]
  z_log = collatz_z(z, add=add, mult=mult, div=div, zs_min=zs_min, ctr_lim=ctr_lim, imag_max=imag_max, real_max=real_max, odd_type=odd_type)
  len_log = len(z_log)
  if len_log > ctr_lim:
   z_min, ccl = cycle_calc_z(z_log)
   if isfinite(z_min.real) and isfinite(z_min.imag):
    zs_min.append(z_min)
    cycles.append(ccl)
 return zs_min, cycles
