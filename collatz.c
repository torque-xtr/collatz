#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <gmp.h>
#include <math.h>
#include <malloc.h>

/* 
ver.250718_1
Added minimum number in collatz_c_short: if iterations fall below it, terminate. Used in jump records calculations.
ver.250623_1
Added glide_init argument in collatz_short for correct glide calculation in c_int -> c_short calls in collatz_c.py
ver.250618_1
Removed mpz_t field in log, 
Added number comparison (cycle checking, glide, n_max) based on byte log (compare_bytes function)
Tested on some 10**6 numbers and different mult, divs, add parameters
ver.250617_1
Added termination check on ns_min only if current number smaller than max(ns_min)
Added correct ctr_o and ctr_e calculation in collatz_short (always excluding last iteration), 
adjusted termination on ctr_lim (len_log == ctr_lim+1)
 */

void free_buffer(unsigned char* buf) {
    free(buf);
}

struct IterLog {
 unsigned char* num_buf;
 size_t* num_offsets;
 size_t buf_capacity;
 int* iter_types;
 size_t ctr;
 size_t ctr_lim;
 size_t cycle_start;
 size_t i_jump;
 size_t i_glide;
 int exit_code;
};


size_t calc_buf_size(int log_n_max, size_t ctr_lim) {
 size_t bits_per_num = (size_t)(log_n_max * 3.33) + 64;
 size_t n_bytes_max = ((bits_per_num * ctr_lim) + 7) / 8; 
 //printf("Buffer size %zu\n", n_bytes_max);
 return n_bytes_max;
 }


void init_iter_log(struct IterLog* iter_log, int log_n_max, size_t ctr_lim) {
 iter_log->ctr_lim = ctr_lim;
 iter_log->num_offsets = (size_t*)malloc((iter_log->ctr_lim+2)*sizeof(size_t)); 
 iter_log->iter_types = (int*)malloc(iter_log->ctr_lim * sizeof(int));
 iter_log->buf_capacity = calc_buf_size(log_n_max, ctr_lim); 
 iter_log->num_buf = (unsigned char*)malloc(iter_log->buf_capacity);
 iter_log->ctr = 0;
 iter_log->cycle_start = 0;
 iter_log->i_jump = 0;
 iter_log->i_glide = 0;
 iter_log->exit_code = -1;

 memset(iter_log->num_offsets, 0, (ctr_lim+2) * sizeof(size_t));
}


//do not rewrite all fields
void reset_iter_log(struct IterLog* iter_log, size_t ctr_lim) {
 iter_log->ctr_lim = ctr_lim;
 iter_log->ctr = 0;
 iter_log->cycle_start = 0;
 iter_log->i_jump = 0;
 iter_log->i_glide = 0;
 iter_log->exit_code = -1;
 iter_log->num_offsets[0] = 0;
}


void free_iter_log(struct IterLog* iter_log){
 if (!iter_log) return;
 size_t ctr_lim = iter_log->ctr_lim;
 if (iter_log->num_buf) free(iter_log->num_buf);
 if (iter_log->num_offsets) free(iter_log->num_offsets);
 if (iter_log->iter_types) free(iter_log->iter_types);
 memset(iter_log, 0, sizeof(struct IterLog));
}

void write_log(struct IterLog* iter_log, mpz_t num) {    
 size_t offset_cur = iter_log->num_offsets[iter_log->ctr];
 size_t count;
 mpz_export(iter_log->num_buf + offset_cur, &count, 1, 1, 0, 0, num);
 offset_cur += count;
 if (offset_cur > iter_log->buf_capacity) {
  fprintf(stderr, "Buffer overflow! Needed %zu, capacity %zu\n", offset_cur, iter_log->buf_capacity);
  exit(1);
  }
 iter_log->num_offsets[iter_log->ctr + 1] = offset_cur; //sentinel
}


void mpz_from_bytes(mpz_t result, const unsigned char* ba, size_t i_start, size_t i_end) {
 size_t length = i_end - i_start;
 if (length < 1) {
  mpz_set_ui(result, 0);
  return;
 }
 mpz_import(result, length, 1, sizeof(unsigned char), 0, 0, ba + i_start);
}

void mpz_set_ull(mpz_t rop, unsigned long long op) {
 mpz_set_ui(rop, (unsigned long)(op >> 32));  // High 32 bits
 mpz_mul_2exp(rop, rop, 32);                 // Shift left 32 bits
 mpz_add_ui(rop, rop, (unsigned long)op);    // Add low 32 bits
}

void num_print(const struct IterLog* iter_log, size_t i_num) {
 mpz_t num;
 mpz_init(num);
 mpz_from_bytes(num, iter_log->num_buf, iter_log->num_offsets[i_num], iter_log->num_offsets[i_num+1]);
 gmp_printf("%Zd ", num); // OK
 mpz_clear(num);
}
// size_t start = iter_log->num_offsets[i_num];
// size_t end = iter_log->num_offsets[i_num + 1];
// size_t num_len = end - start;



struct CollatzResults {
 unsigned char* n_last_buf;
 size_t n_last_size;
 unsigned char* jump_buf;
 size_t jump_size;
 size_t ctr;
 size_t* op_ctrs;
 size_t op_ctrs_size;
 size_t glide;
};

void init_collatz_result(struct CollatzResults* res, size_t n_divs) {
 memset(res, 0, sizeof(struct CollatzResults));
 res->op_ctrs_size = n_divs + 1;
 res->op_ctrs = (size_t*)malloc((res->op_ctrs_size)*sizeof(size_t));
 memset(res->op_ctrs, 0, (n_divs+1) * sizeof(size_t));
}


void free_collatz_result(struct CollatzResults* res) {
 if (res->n_last_buf) {
  void (*freefunc)(void *, size_t);
  mp_get_memory_functions(NULL, NULL, &freefunc);
  freefunc(res->n_last_buf, res->n_last_size + 1);
 }

 if (res->jump_buf) {
  void (*freefunc)(void *, size_t);
  mp_get_memory_functions(NULL, NULL, &freefunc);
  freefunc(res->jump_buf, res->jump_size + 1);
 }

 if (res->op_ctrs) {
  void (*freefunc)(void *, size_t);
  mp_get_memory_functions(NULL, NULL, &freefunc);
  freefunc(res->op_ctrs, res->op_ctrs_size + 1);
 }

 memset(res, 0, sizeof(struct CollatzResults));
}

struct CR_Int {
 unsigned long long n_last;
 unsigned long long jump;
 size_t ctr;
 size_t* op_ctrs;
 size_t op_ctrs_size;
 size_t glide;
};

void init_cr_int(struct CR_Int* res, size_t n_divs) {
 memset(res, 0, sizeof(struct CR_Int));
 res->op_ctrs_size = n_divs + 1;
 res->op_ctrs = (size_t*)malloc((res->op_ctrs_size)*sizeof(size_t));
 memset(res->op_ctrs, 0, (res->op_ctrs_size) * sizeof(size_t));
}


void free_cr_int(struct CR_Int* res) {
 if (res->op_ctrs) {
  void (*freefunc)(void *, size_t);
  mp_get_memory_functions(NULL, NULL, &freefunc);
  freefunc(res->op_ctrs, res->op_ctrs_size + 1);
 }
 memset(res, 0, sizeof(struct CR_Int));
 //free(res);
}

/*
void collatz_iter(mpz_t num, const mpz_t mult, const mpz_t* divs, const mpz_t add, size_t div_count, struct CollatzResults* res) {
 if (!num || !mult || !divs || !add){
  return -1;
 } 
 bool divided = false;
 for (size_t i = 0; i < div_count; i++) {
  if (mpz_divisible_p(num, divs[i]) != 0) {
   mpz_divexact(num, num, divs[i]);
   res->op_ctrs[i]++;
   divided = true;
  }
 }
 if !divided {
  mpz_mul(num, num, mult);
  mpz_add(num, num, add);
  res->op_ctrs[0]++;
  res->ctr++;
 }
}

*/
int compare_bytes(const struct IterLog* iter_log, size_t i_1, size_t i_2) {
 size_t offset_1_st = iter_log->num_offsets[i_1];
 size_t offset_1_en = iter_log->num_offsets[i_1+1];
 size_t offset_2_st = iter_log->num_offsets[i_2];
 size_t offset_2_en = iter_log->num_offsets[i_2+1];
 
 size_t len1 = offset_1_en - offset_1_st;
 size_t len2 = offset_2_en - offset_2_st;
 
 int cmp_val = 0; 
 bool cont = true;
 // Compare lengths 
 if (len1 < len2) {
   cmp_val = -1;
 } else if (len1 > len2) {
   cmp_val = 1;
 } else { // Compare byte-by-byte from most significant end

  const unsigned char* p1 = iter_log->num_buf + offset_1_st;
  const unsigned char* p2 = iter_log->num_buf + offset_2_st;
 
  for (size_t i = 0; i < len1; i++) {
   //printf("num bytes compared %d %d\n", *p1, *p2);
   if (p1[i] < p2[i]) {
    cmp_val = -1;
    break;
   } else if (p1[i] > p2[i]) {
    cmp_val = 1;
    break;
   }
  }
 }
 //printf("cmp val %d\n", cmp_val);
 return cmp_val; // all bytes are equal
}


size_t cycle_check(struct IterLog* iter_log, size_t ctr_l) {
 iter_log->cycle_start = iter_log->ctr-1;
 size_t ctr_min = iter_log->ctr - 1;

 //checking for n_min
 bool cycle_detected = false;
 for (size_t i=iter_log->ctr-2; i > ctr_l; i--) {
  if (compare_bytes(iter_log, i, ctr_min) <= 0) {
   ctr_min = i;
  }
  if (compare_bytes(iter_log, i, iter_log->ctr-1) == 0) {
   cycle_detected = true;
   iter_log->cycle_start = i;
   break;
  }
 }

 if (cycle_detected) {
  for (size_t i=0; i < iter_log->ctr; i++) {
   if (compare_bytes(iter_log, i, ctr_min) == 0) {
    iter_log->cycle_start = i;
    break;
   }
  }
 }
 //printf("ctr, ctr min, ctr start, detected, %zu %zu %zu %d \n", iter_log->ctr, ctr_min, iter_log->cycle_start, (int)cycle_detected);
 return ctr_min;
}


void collatz(
 const unsigned char* n_0_buf, size_t n_0_size,
 int multiplier,
 const int* divisors, size_t div_count,
 int addition,
 int log_n_max,
 size_t n_iter_max,
 const int* end_vals, size_t ns_min_count,
 struct IterLog* iter_log
)
{
 
 //malloc_stats();
 
 if (!n_0_buf || !divisors || !end_vals || !iter_log) {
  fprintf(stderr, "NULL pointer argument\n");
  return;
 }
 
 //initialize collatz parameters
 mpz_t n, mult, add, n_max, n_max_base, max_ns_min;
 mpz_inits(n, mult, add, n_max, n_max_base, max_ns_min, NULL);
 mpz_set_ui(n_max_base, 10);
 
 mpz_import(n, n_0_size, 1, 1, 0, 0, n_0_buf);
 
 mpz_set_si(mult, multiplier);
 mpz_set_si(add, addition);

 mpz_pow_ui(n_max, n_max_base, (unsigned int)log_n_max);
 
 mpz_t *divs=NULL;
 divs = (mpz_t*)malloc(div_count * sizeof(mpz_t));
 if (!divs) goto cleanup;
 for (size_t i=0; i < div_count; i++) {
  mpz_init(divs[i]);
  mpz_set_si(divs[i], divisors[i]);
 }

 mpz_t *ns_min=NULL;
 ns_min = (mpz_t*)malloc((ns_min_count) * sizeof(mpz_t));
 if (!ns_min) goto cleanup;
 for (size_t i=0; i < ns_min_count; i++) {
  mpz_init(ns_min[i]);
  mpz_set_si(ns_min[i], end_vals[i]);
 }

 mpz_set_ui(max_ns_min, 0); //calculate max of end values to speed-up checks
 for (size_t i=0; i < ns_min_count; i++) {
  if (mpz_cmp(max_ns_min, ns_min[i]) <= 0) {
   mpz_set(max_ns_min, ns_min[i]);  
  }
 }

 //gmp_printf("max n_min: %Zd\n", max_ns_min); // OK
 //preparing array of control ctr values
 size_t ctr_check_max = (size_t)(round(cbrt(n_iter_max)) + 3);
 
 size_t *ctrs_check = NULL;
 ctrs_check = (size_t*)malloc(ctr_check_max * sizeof(size_t));
 for (size_t i=0; i < ctr_check_max-1; i++) {
  ctrs_check[i] = (size_t)(i*i*i);
 }
 ctrs_check[ctr_check_max-1] = (size_t)n_iter_max-1;
 size_t ctr_check = 2;
 
 //---------------------------------------------------------------
 //---------------------main cycle--------------------------------
 //---------------------------------------------------------------
 
 int iter_type = 0; 
 write_log(iter_log, n);
 while(iter_log->ctr < n_iter_max) {
  //collatz iteration //later, rewrite with 'while divisible' cycle, so that no divisibility checks are repeated (may be much faster?)
  bool divided = false;
  for (size_t i = 0; i < div_count; i++) {
   if (!divided && mpz_divisible_p(n, divs[i]) != 0) {
    mpz_divexact(n, n, divs[i]);
    iter_log->iter_types[iter_log->ctr] = mpz_get_si(divs[i]);
    divided = true;
    break;
   }
  }
  if (!divided) {
   mpz_mul(n, n, mult);
   mpz_add(n, n, add);
   iter_log->iter_types[iter_log->ctr] = 0;
  }

  iter_log->ctr++;
  write_log(iter_log, n);
   
 //termination conditions check
  bool terminate = false;
  if (mpz_cmp(n, n_max) > 0) {
   iter_log->exit_code = 2; 
   terminate = true; //check if current iteration is above limit
   }
 
  //starting number == current number: was inside a cycle
  if (compare_bytes(iter_log, 0, iter_log->ctr) == 0) {
   terminate = true;
   iter_log->exit_code = 4;
  }
  
 //check if one of end numbers is reached
  if (!terminate && iter_log->ctr > 0 && mpz_cmp(n, max_ns_min) <= 0) {
   for (size_t i = 0; i < ns_min_count; i++) {
    if (mpz_cmp(n, ns_min[i]) == 0) {//     gmp_printf("terminated at n_min: %Zd", ns_min[i]); //     printf(" ctr %d\n", iter_log->ctr); // OK
     terminate = true;
     iter_log->exit_code = 0; 
     break;
    }
   }
  }
   
  //check if sequence has entered a cycle;
  //if cycle detected, set ctr to the lowest cycle member ('truncate' iter_log) and terminate
  if (iter_log->ctr == ctrs_check[ctr_check] && iter_log->exit_code != 4) {
   
   size_t ctr_l = ctr_check % 2 == 0 ? ctrs_check[ctr_check-1] : 1;
   ctr_check += 1;
   size_t ctr_min = cycle_check(iter_log, ctr_l);
   if (iter_log->cycle_start != iter_log->ctr-1) {
    iter_log->ctr = iter_log->cycle_start; 
    terminate=true;
    iter_log->exit_code = 1;
    mpz_from_bytes(n, iter_log->num_buf, iter_log->num_offsets[iter_log->ctr], iter_log->num_offsets[iter_log->ctr+1]);
   }
  }
 
  //gmp_printf("%Zd ", n); // OK
  //num_print(iter_log, iter_log->ctr);
  //printf(" (ctr = %zu) \n", iter_log->ctr);
  if (terminate) break;
 }
 
 iter_log->iter_types[iter_log->ctr] = 0; // writing final iteration type 
 for (size_t i = 0; i < div_count; i++) {
  if (mpz_divisible_p(n, divs[i]) != 0) {
   iter_log->iter_types[iter_log->ctr] = mpz_get_si(divs[i]);
   break;
  }
 }

 //stats calculation
 bool glide_det = true;
 for (size_t i=0; i < iter_log->ctr+1; i++) {
  int cb = compare_bytes(iter_log, i, iter_log->i_jump);
//  num_print(iter_log, i);
//  printf(" %d ", cb);
//  num_print(iter_log, iter_log->i_jump);
//  printf("\n");
  if (cb > 0) iter_log->i_jump = i;
  if (glide_det && (compare_bytes(iter_log, i, 0) < 0))  {
    iter_log->i_glide = i;
    glide_det = false;
   }
 }
 if (glide_det) iter_log->i_glide = iter_log->ctr;
 
 if (iter_log->ctr >= n_iter_max) iter_log->exit_code = 3; 

// gmp_printf("%Zd ", n); // OK
// num_print(iter_log, iter_log->ctr);
// printf(" (ctr = %zu) \n", iter_log->ctr);
 
 cleanup: 
  if (divs) {
   for (size_t i=0; i < div_count; i++) mpz_clear(divs[i]);
   free(divs);
  }
 
  if (ns_min) {
   for (size_t i=0; i < ns_min_count; i++) mpz_clear(ns_min[i]);
   free(ns_min);
  }
  
  free(ctrs_check);
  
  mpz_clears(n, mult, add, n_max, n_max_base, max_ns_min, NULL);
 //malloc_stats();
}

 
 
//====================================================================================================== 
//====================================faster version without full log writing===========================
//======================================================================================================

void collatz_short(
 const unsigned char* n_0_buf, size_t n_0_size,
 int multiplier,
 const int* divisors, size_t div_count,
 int addition,
 int log_n_max,
 size_t n_iter_max,
 const int* end_vals, size_t ns_min_count,
 unsigned long long int glide_init,
 unsigned long long int n_min_lim_int,
 struct CollatzResults* res
)
{
 
 if (!n_0_buf || !divisors || !end_vals || !res) {
  fprintf(stderr, "NULL pointer argument\n");
  return;
 }
 
 //initialize collatz parameters
 mpz_t n, mult, add, n_max, n_max_base, max_ns_min, n_start, jump, n_check, n_min, n_min_lim, n_0_init;
 mpz_inits(n, mult, add, n_max, n_max_base, max_ns_min, n_start, jump, n_check, n_min, n_min_lim, n_0_init, NULL);
 mpz_set_ui(n_max_base, 10);
 
 mpz_import(n, n_0_size, 1, 1, 0, 0, n_0_buf);
 
 mpz_set_si(mult, multiplier);
 mpz_set_si(add, addition);

 mpz_pow_ui(n_max, n_max_base, (unsigned int)log_n_max);
 
 mpz_t *divs=NULL;
 divs = (mpz_t*)malloc(div_count * sizeof(mpz_t));
 if (!divs) goto cleanup;
 for (size_t i=0; i < div_count; i++) {
  mpz_init(divs[i]);
  mpz_set_si(divs[i], divisors[i]);
 }

 mpz_t *ns_min=NULL;
 ns_min = (mpz_t*)malloc((ns_min_count) * sizeof(mpz_t));
 if (!ns_min) goto cleanup;
 for (size_t i=0; i < ns_min_count; i++) {
  mpz_init(ns_min[i]);
  mpz_set_si(ns_min[i], end_vals[i]);
 }

 mpz_set_ui(max_ns_min, 0); //calculate max of end values to speed-up checks
 for (size_t i=0; i < ns_min_count; i++) {
  if (mpz_cmp(max_ns_min, ns_min[i]) <= 0) {
   mpz_set(max_ns_min, ns_min[i]);  
  }
 }

 mpz_set(jump, n);
 mpz_set(n_start, n);
 mpz_set(n_check, n);
 mpz_set(n_min, n);
 
 if (glide_init <= 0) { 
  mpz_set(n_0_init, n_start);
 } else {
  mpz_set_ull(n_0_init, glide_init); //transfer initial number for c_int -> c_short calculation
 }

 mpz_set_ull(n_min_lim, n_min_lim_int); //transfer initial number for c_int -> c_short calculation

 //gmp_printf("n_0_init %Zd\n", n_0_init);
 size_t ctr_check = 100; 
 bool cycle_found = false;
 //---------------------------------------------------------------
 //---------------------main cycle--------------------------------
 //---------------------------------------------------------------
 
 int iter_type = 0; 
 bool glide_det = true;
 while(res->ctr < n_iter_max) {

  //collatz iteration //later, rewrite with 'while divisible' cycle, so that no divisibility checks are repeated (may be much faster?)
  bool divided = false;
  for (size_t i = 0; i < div_count; i++) {
   if (mpz_divisible_p(n, divs[i]) != 0) {
    mpz_divexact(n, n, divs[i]);
    res->op_ctrs[i+1]++;
    divided = true;
    break;
   }
  }
  if (!divided) {
   mpz_mul(n, n, mult);
   mpz_add(n, n, add);
   res->op_ctrs[0]++;
  }
 res->ctr++;

 //jump and glide tracking
 if (mpz_cmp(n, jump) > 0) mpz_set(jump, n);
 if (glide_det) {
  if (mpz_cmp(n, n_0_init) < 0) {
   res->glide = res->ctr;
   glide_det = false;
  }
 }

 //termination conditions check
  bool terminate = false;
  if (mpz_cmp(n, n_min_lim) <= 0) terminate = true; //check if current iteration is below lower limiting number (usually zero; used in fast jump records calculation)
  if (mpz_cmp(n, n_max) > 0) terminate = true; //check if current iteration is above limit
  if (mpz_cmp(n, n_start) == 0) terminate = true;
 
 //check if one of end numbers is reached (initial number is added here)
  if (res->ctr > 0 && mpz_cmp(n, max_ns_min) <= 0) {
   for (size_t i = 0; i < ns_min_count; i++) {
    if (mpz_cmp(n, ns_min[i]) == 0) {   //gmp_printf("terminated at n_min: %Zd", ns_min[i]); //     //printf(" ctr %d\n", res->ctr); // OK
     terminate = true;
     break;
    }
   }
  }

 //cycle checking
 //if a new cycle is detected, stats are wrong because at least some iterations are within cycle
 //in this case, calculate it externally and re-launch function with new n_min found to obtain correct stats
 if (cycle_found) {
  int cmp_val = mpz_cmp(n, n_min);
  if (cmp_val < 0) mpz_set(n_min, n);
  if (cmp_val == 0) terminate = true; // this happens if a starting number is anywhere inside the cycle, not just n == cycle_min
 }

 if (!cycle_found && mpz_cmp(n, n_check) == 0) {
  cycle_found = true;
  mpz_set(n_min, n);
 }

 if (!cycle_found && res->ctr == ctr_check) {
  mpz_set(n_check, n);
  ctr_check = ctr_check * 3 / 2;
 }

 if (terminate) break;
 }
   
 //writing results
 res->n_last_buf = mpz_export(NULL, &res->n_last_size, 1, 1, 0, 0, n);
 res->jump_buf = mpz_export(NULL, &res->jump_size, 1, 1, 0, 0, jump);
  
 if (glide_det) res->glide = res->ctr;

 cleanup: 
  if (divs) {
   for (size_t i=0; i < div_count; i++) mpz_clear(divs[i]);
   free(divs);
  }
 
  if (ns_min) {
   for (size_t i=0; i < ns_min_count; i++) mpz_clear(ns_min[i]);
   free(ns_min);
  }

  mpz_clears(n, mult, add, n_max, n_max_base, max_ns_min, n_start, jump, n_check, n_min, n_min_lim, n_0_init, NULL);
 
}
 




//====================================================================================================== 
//==========ulonglong version for 3n+1 and other convergent iter types on small numbers ================
//======================================================================================================

void collatz_int(
 unsigned long long int n_0,
 unsigned long long int mult,
 unsigned long long int* divisors, size_t div_count,
 unsigned long long int add,
 unsigned long long int n_max,
 size_t n_iter_max,
 unsigned long long int* ns_min, size_t ns_min_count,
 struct CR_Int* res
)
{
 
 if (!n_0 || !divisors || !ns_min || !res) {
  fprintf(stderr, "some error in arguments\n");
  return;
 }
 //printf("check\n");

 //initialize collatz parameters
 unsigned long long int num = n_0, n_start = n_0, n_check = n_0, n_min = n_0;
 unsigned long long int max_ns_min = 0;
 res->jump = n_0;
 
 for (size_t i=0; i < ns_min_count; i++) { //calculate max of end values to speed-up checks
  if (max_ns_min < ns_min[i]) max_ns_min = ns_min[i];  
 }
 
 size_t ctr_check = 20; 
 bool cycle_found = false;
 //---------------------------------------------------------------
 //---------------------main cycle--------------------------------
 //---------------------------------------------------------------
 
 int iter_type = 0; 
 bool glide_det = true;
 while(res->ctr < n_iter_max) {

  //collatz iteration
  bool divided = false;
  for (size_t i = 0; i < div_count; i++) {//   printf("number, ctr, div, res_div %d %d %d %d\n", num, res->ctr, divisors[i], num % divisors[i]);
   if (num % divisors[i] == 0) {
    num = num / divisors[i];
    res->op_ctrs[i+1] += 1;
    divided = true;
    break;
   }
 }
 if (!divided) {
  num = num * mult + add;
  res -> op_ctrs[0] += 1;
 }

 res->ctr++;

 //jump and glide tracking
 
 if (num > res->jump) res->jump = num;
 if (glide_det && num < n_start) {
   res->glide = res->ctr;
   glide_det = false;
  }

 //termination conditions check
  bool terminate = false;
  if (num > n_max) terminate = true; //check if current iteration is above limit
  if (num == n_start) terminate = true;
 
 //check if one of end numbers is reached (initial number is added here)
  if (res->ctr > 0 && num <= max_ns_min) {
   for (size_t i = 0; i < ns_min_count; i++) {
    if (num == ns_min[i]) {   
     //printf("terminated at n_min: %d", ns_min[i]); 
     //printf(" ctr %d\n", res->ctr); // OK
     terminate = true;
     break;
    }
   }
  }

 //cycle checking
 //if a new cycle is detected, stats are wrong because at least some iterations are within cycle
 //in this case, calculate it externally and re-launch function with new n_min found to obtain correct stats
 
 //if (cycle_found) printf("current ctr, n, n_min: %d %d %d\n", res->ctr, num, n_min); 

 if (cycle_found) {
  if (num == n_min) terminate = true;
  else if (num < n_min) n_min = num;
 }
 
 if (!cycle_found && num == n_check) {
  n_min = num;
  cycle_found = true;
 }

 if (!cycle_found && res->ctr == ctr_check) {
  n_check = num;
  ctr_check = ctr_check * 3 / 2;
 }

 if (terminate) break;
 }
 
 //writing results
 res->n_last = num;;  
 if (glide_det) res->glide = res->ctr;
 
// cleanup: 
//  free(divisors);
//  free(ns_min);

}
