#include <nmmintrin.h>

void sgemm( int m, int n, int d, float *A, float *C )
{
  __m128 a2_1, a2_2, a2_3, a2_4,a2_5, a2_6,a1, c;
  if(n == 40 && m == 48 ){
      
    for( int k = 0; k < m; k++ ){ 
      for( int j = 0; j < n; j+=4 ){
	 a2_1 = _mm_load1_ps(A+j*(n+1)+k*(n));
	 a2_2 = _mm_load1_ps(A+(j+1)*(n+1)+k*(n));
	 a2_3 = _mm_load1_ps(A+(j+2)*(n+1)+k*(n));
	 a2_4 = _mm_load1_ps(A+(j+3)*(n+1)+k*(n));
	 for( int i = 0; i < n; i+=8 ){
	  
	  a1 = _mm_loadu_ps(A+i+k*(n)); 
	  
	  c = _mm_loadu_ps(C+i+j*(n));
	  c = _mm_add_ps(c, _mm_mul_ps(a1, a2_1));
	  _mm_storeu_ps(C+i+j*(n), c);
	  
	  c = _mm_loadu_ps(C+i+(j+1)*(n));
	  c = _mm_add_ps(c, _mm_mul_ps(a1, a2_2));
	  _mm_storeu_ps(C+i+(j+1)*(n), c);
	  
	  c = _mm_loadu_ps(C+i+(j+2)*(n));
	  c = _mm_add_ps(c, _mm_mul_ps(a1, a2_3));
	  _mm_storeu_ps(C+i+(j+2)*(n), c);
	  
	  c = _mm_loadu_ps(C+i+(j+3)*(n));
	  c = _mm_add_ps(c, _mm_mul_ps(a1, a2_4));
	  _mm_storeu_ps(C+i+(j+3)*(n), c);
	  
	  
	  a1 = _mm_loadu_ps(A+i+4+k*(n)); 
	  c = _mm_loadu_ps(C+i+4+j*(n));
	  c = _mm_add_ps(c, _mm_mul_ps(a1, a2_1));
	  _mm_storeu_ps(C+i+4+j*(n), c);
	   
	  c = _mm_loadu_ps(C+i+4+(j+1)*(n));
	  c = _mm_add_ps(c, _mm_mul_ps(a1, a2_2));
	  _mm_storeu_ps(C+i+4+(j+1)*(n), c);
	  
	  c = _mm_loadu_ps(C+i+4+(j+2)*(n));
	  c = _mm_add_ps(c, _mm_mul_ps(a1, a2_3));
	  _mm_storeu_ps(C+i+4+(j+2)*(n), c);
	  
	  c = _mm_loadu_ps(C+i+4+(j+3)*(n));
	  c = _mm_add_ps(c, _mm_mul_ps(a1, a2_4));
	  _mm_storeu_ps(C+i+4+(j+3)*(n), c);
	  
	 }
	  
      }
    }

  } else { 

  }
}
