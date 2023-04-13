
/* Create macros so that the matrices are stored in column-major order */

#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]

/* Routine for computing C = A * B + C */

void AddDot4x4( int, double *, int, double *, int, double *, int );

void MY_MMult( int m, int n, int k, double *a, int lda, 
                                    double *b, int ldb,
                                    double *c, int ldc )
{
  int i, j;

  for ( j=0; j<n; j+=4 ){        /* Loop over the columns of C, unrolled by 4 */
    for ( i=0; i<m; i+=4 ){        /* Loop over the rows of C */
      /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
	 one routine (four inner products) */

      AddDot4x4( k, &A( i,0 ), lda, &B( 0,j ), ldb, &C( i,j ), ldc );
    }
  }
}

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3
#include <immintrin.h>

typedef union
{
  __m256d v;
  double d[4];
} v4df_t;

void AddDot4x4( int k, double *a, int lda,  double *b, int ldb, double *c, int ldc )
{
  /* So, this routine computes a 4x4 block of matrix A

           C( 0, 0 ), C( 0, 1 ), C( 0, 2 ), C( 0, 3 ).  
           C( 1, 0 ), C( 1, 1 ), C( 1, 2 ), C( 1, 3 ).  
           C( 2, 0 ), C( 2, 1 ), C( 2, 2 ), C( 2, 3 ).  
           C( 3, 0 ), C( 3, 1 ), C( 3, 2 ), C( 3, 3 ).  

     Notice that this routine is called with c = C( i, j ) in the
     previous routine, so these are actually the elements 

           C( i  , j ), C( i  , j+1 ), C( i  , j+2 ), C( i  , j+3 ) 
           C( i+1, j ), C( i+1, j+1 ), C( i+1, j+2 ), C( i+1, j+3 ) 
           C( i+2, j ), C( i+2, j+1 ), C( i+2, j+2 ), C( i+2, j+3 ) 
           C( i+3, j ), C( i+3, j+1 ), C( i+3, j+2 ), C( i+3, j+3 ) 
	  
     in the original matrix C 

     And now we use vector registers and instructions */

  int p;

  __m256d
    c_00_c_30_vreg,    c_01_c_31_vreg,    c_02_c_32_vreg,    c_03_c_33_vreg,
    a_0p_a_3p_vreg,
    b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg; 

  double 
    /* Point to the current elements in the four columns of B */
    *b_p0_pntr, *b_p1_pntr, *b_p2_pntr, *b_p3_pntr; 
    
  b_p0_pntr = &B( 0, 0 );
  b_p1_pntr = &B( 0, 1 );
  b_p2_pntr = &B( 0, 2 );
  b_p3_pntr = &B( 0, 3 );

  c_00_c_30_vreg = _mm256_setzero_pd();
  c_01_c_31_vreg = _mm256_setzero_pd();
  c_02_c_32_vreg = _mm256_setzero_pd();
  c_03_c_33_vreg = _mm256_setzero_pd();

  for ( p=0; p<k; p++ ){
      a_0p_a_3p_vreg = _mm256_loadu_pd( (double *) &A( 0, p ) );

    b_p0_vreg = _mm256_broadcast_sd( (double *) b_p0_pntr++ );   /* load and duplicate */
    b_p1_vreg = _mm256_broadcast_sd( (double *) b_p1_pntr++ );   /* load and duplicate */
    b_p2_vreg = _mm256_broadcast_sd( (double *) b_p2_pntr++ );   /* load and duplicate */
    b_p3_vreg = _mm256_broadcast_sd( (double *) b_p3_pntr++ );   /* load and duplicate */

    /* First row to fourth rows */
    c_00_c_30_vreg = _mm256_add_pd(c_00_c_30_vreg, _mm256_mul_pd(a_0p_a_3p_vreg, b_p0_vreg));
    c_01_c_31_vreg = _mm256_add_pd(c_01_c_31_vreg, _mm256_mul_pd(a_0p_a_3p_vreg, b_p1_vreg));
    c_02_c_32_vreg = _mm256_add_pd(c_02_c_32_vreg, _mm256_mul_pd(a_0p_a_3p_vreg, b_p2_vreg));
    c_03_c_33_vreg = _mm256_add_pd(c_03_c_33_vreg, _mm256_mul_pd(a_0p_a_3p_vreg, b_p3_vreg));
  }

  _mm256_store_pd( &C( 0, 0 ), _mm256_add_pd( _mm256_loadu_pd( (double *) &C( 0, 0 ) ), c_00_c_30_vreg ));
  _mm256_store_pd( &C( 0, 1 ), _mm256_add_pd( _mm256_loadu_pd( (double *) &C( 0, 1 ) ), c_01_c_31_vreg ));
  _mm256_store_pd( &C( 0, 2 ), _mm256_add_pd( _mm256_loadu_pd( (double *) &C( 0, 2 ) ), c_02_c_32_vreg ));
  _mm256_store_pd( &C( 0, 3 ), _mm256_add_pd( _mm256_loadu_pd( (double *) &C( 0, 3 ) ), c_03_c_33_vreg ));
}
