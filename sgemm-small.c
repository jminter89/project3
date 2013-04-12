#include <nmmintrin.h>

void sgemm( int m, int n, int d, float *A, float *C )
{
	/*if(n == 40 && m == 48 ){
  		for( int i = 0; i < n; i++ ){
    		for( int k = 0; k < m; k++ ){
    			__m128 a1 = _mm_loadu_ps(A+i+k*(n));  
      			for( int j = 0; j < n; j++ ){
      				__m128 c = _mm_loadu_pd(C+i+j*(n));
      				__m128 a2 = _mm_load1_ps(A+j*(n+1)+k*(n));

      				c = _mm_add_pd(c, _mm_mul_pd(a1, a2));
      				_mm_storeu_pd(C+i+j*(n)), c);


					/*cVector = _mm_add_ps(cVector, aVect0, aVect1);
					C[i+0*n] += (float*)cVector[0];
					C[i+1*n] += (float*)cVector[1]; 
					C[i+2*n] += (float*)cVector[2];
					C[i+3*n] += (float*)cVector[3];
				}
			}
		}
	} else { */
  		for( int i = 0; i < n; i++ ){
    		for( int k = 0; k < m; k++ ){  
      			for( int j = 0; j < n; j++ ){
					C[i+j*n] += A[i+k*(n)] * A[j*(n+1)+k*(n)];
				}
			}
		}
		
}
