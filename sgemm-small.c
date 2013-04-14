#include <nmmintrin.h>
#include <string.h>
#include <stdio.h>

#define STRIDE 8

void sgemm( int m, int n, int d, float *A, float *C )
{
  
  if(n == 40 && m == 48 ){
    
    __m128 a2_1, a2_2, a2_3, a2_4,a2_5, a2_6, a1, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8;
    register int k,j,i;
      
    for( k = 0; k < m; k++ ){ 
      for( j = 0; j < n; j+=4 ){
	 a2_1 = _mm_load1_ps(A+j*(n+1)+k*(n));
	 a2_2 = _mm_load1_ps(A+(j+1)*(n+1)+k*(n));
	 a2_3 = _mm_load1_ps(A+(j+2)*(n+1)+k*(n));
	 a2_4 = _mm_load1_ps(A+(j+3)*(n+1)+k*(n));
	 for( i = 0; i < n; i+=8 ){
	  
	  a1 = _mm_loadu_ps(A+i+k*(n)); 
	  
	  c_1 = _mm_loadu_ps(C+i+j*(n));
	  c_1 = _mm_add_ps(c_1, _mm_mul_ps(a1, a2_1));
	  
	  
	  c_2 = _mm_loadu_ps(C+i+(j+1)*(n));
	  c_2 = _mm_add_ps(c_2, _mm_mul_ps(a1, a2_2));
	  _mm_storeu_ps(C+i+j*(n), c_1);
	  _mm_storeu_ps(C+i+(j+1)*(n), c_2);
	  
	  c_3 = _mm_loadu_ps(C+i+(j+2)*(n));
	  c_3 = _mm_add_ps(c_3, _mm_mul_ps(a1, a2_3));
	  
	  
	  c_4 = _mm_loadu_ps(C+i+(j+3)*(n));
	  c_4 = _mm_add_ps(c_4, _mm_mul_ps(a1, a2_4));
	  _mm_storeu_ps(C+i+(j+2)*(n), c_3);
	  _mm_storeu_ps(C+i+(j+3)*(n), c_4);
	  
	  
	  a1 = _mm_loadu_ps(A+i+4+k*(n));
	  
	  c_5 = _mm_loadu_ps(C+i+4+j*(n));
	  c_5 = _mm_add_ps(c_5, _mm_mul_ps(a1, a2_1));
	  
	   
	  c_6 = _mm_loadu_ps(C+i+4+(j+1)*(n));
	  c_6 = _mm_add_ps(c_6, _mm_mul_ps(a1, a2_2));
	  _mm_storeu_ps(C+i+4+j*(n), c_5);
	  _mm_storeu_ps(C+i+4+(j+1)*(n), c_6);
	  
	  c_7 = _mm_loadu_ps(C+i+4+(j+2)*(n));
	  c_7 = _mm_add_ps(c_7, _mm_mul_ps(a1, a2_3));
	  
	  
	  c_8 = _mm_loadu_ps(C+i+4+(j+3)*(n));
	  c_8 = _mm_add_ps(c_8, _mm_mul_ps(a1, a2_4));
	  _mm_storeu_ps(C+i+4+(j+2)*(n), c_7);
	  _mm_storeu_ps(C+i+4+(j+3)*(n), c_8);
	  
	 }
	  
      }
    }

  } else {
    __m128 a2_1, a2_2, a1, c_1, c_2, c_3, c_4;
    register int k,j,i;
    
    const int addRow = n - (n%STRIDE);
   	const int padRow = addRow + STRIDE;
    const int padSizeA = padRow * (d+n);
    const int padSizeC = padRow * n;
    float Apad[padSizeA];
    float Cpad[padSizeC];
    
    for(j = 0; j < (d+n); j++){
      	memcpy((float*)(Apad + padRow*j), A + n*j, n*sizeof(float));
      	memset((float*)(Apad + padRow*j+n), 0, (padRow-n)*sizeof(float));
    }

   
    for(j = 0; j < n; j++){	
    	memcpy((float*)(Cpad + padRow*j), C + n*j, n*sizeof(float));
    	memset((float*)(Cpad + padRow*j + n), 0, (padRow - n)*sizeof(float));
    }
    
    for( k = 0; k < m; k++ ){ 
    	for( j = 0; j < n; j+=2 ){
	  		a2_1 = _mm_load1_ps(Apad+j*(padRow+1)+k*(padRow));
	  		a2_2 = _mm_load1_ps(Apad+(j+1)*(padRow+1)+k*(padRow));
		 		for( i = 0; i < n; i+=STRIDE ){
	  
	  			a1 = _mm_loadu_ps(Apad+i+k*(padRow)); 
	  
	  			c_1 = _mm_loadu_ps(Cpad+i+j*(padRow));
	  			c_1 = _mm_add_ps(c_1, _mm_mul_ps(a1, a2_1));

	  			c_2 = _mm_loadu_ps(Cpad+i+(j+1)*(padRow));
	  			c_2 = _mm_add_ps(c_2, _mm_mul_ps(a1, a2_2));

	  			_mm_storeu_ps(Cpad+i+j*(padRow), c_1);
	  			_mm_storeu_ps(Cpad+i+(j+1)*(padRow), c_2);

	  			a1 = _mm_loadu_ps(Apad+i+4+k*(padRow)); 
	  
	  			c_3= _mm_loadu_ps(Cpad+i+4+j*(padRow));
	  			c_3 = _mm_add_ps(c_3, _mm_mul_ps(a1, a2_1));

	  			c_4= _mm_loadu_ps(Cpad+i+4+(j+1)*(padRow));
	  			c_4 = _mm_add_ps(c_4, _mm_mul_ps(a1, a2_2));

	  			_mm_storeu_ps(Cpad+i+4+j*(padRow), c_3);
	  			_mm_storeu_ps(Cpad+i+4+(j+1)*(padRow), c_4);
	 		}
      	}
    }

    for(j = 0; j < n; j++){	
    	memcpy((float*)(C + n*j), Cpad + j*padRow, n*sizeof(float));
    }
  }
}
