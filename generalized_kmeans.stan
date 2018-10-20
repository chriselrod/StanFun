functions {
  matrix tri_inverse(matrix L){
    int N = rows(L);
    matrix[N,N] Li;
    real x;
   
    for (n in 1:N){
      Li[n,n] = 1/L[n,n];
      for (n2 in n+1:N){
        Li[n,n2] = 0;
      }
    }
    for (n2 in 1:(N-1)){ // row_displacement
      for (n1 in 1:(N-n2)){ // column
        x = Li[n1+n2,n1+1] * L[n1+1,n1];
        for (n3 in (n1+2):(n1+n2)){
          x += Li[n1+n2,n3] * L[n3,n1];
        }
        Li[n1+n2,n1] = -Li[n1,n1] * x;
      }
    }
    return Li;
  }
  real normal_mixture_lpdf(matrix y, vector lambda, matrix mu, matrix[] L) {
    int N = rows(y);
    int K = num_elements(lambda);
    real log_density = 0.0;
    vector[K] loglambda = log(lambda);
    vector[K] theta;
    
    for (n in 1:N){
      for (k in 1:K){
        theta[k] = loglambda[k] + multi_normal_cholesky_lpdf(y[n] | mu[k], L[k]);
      }
      log_density += log_sum_exp(theta);
    }
    return log_density;
  }
  real normal_mixture_fast_lpdf(matrix y, vector lambda, matrix mu, matrix[] L) {
    int N = rows(y);
    int D = cols(y);
    int K = num_elements(lambda);
    real log_density = 0.0;
    matrix[D,D] Li[K];
    vector[K] loglambda = log(lambda);
    vector[K] theta;
    
    for (k in 1:K){
      Li[k] = inverse(L[k]);
      // adding determinant of the normal.
      loglambda[k] += -0.5*D*log(2*pi());
      for (d in 1:D){
        loglambda[k] += log(Li[k][d,d]);
      }
    }
    
    for (n in 1:N){
      for (k in 1:K){
        theta[k] = loglambda[k] - 0.5 * dot_self(Li[k] * (y[n] - mu[k])');
      }
      log_density += log_sum_exp(theta);
    }
    return log_density;
  }
  real normal_mixture_fast2_lpdf(matrix y, vector lambda, matrix mu, matrix[] L) {
    int N = rows(y);
    int D = cols(y);
    int K = num_elements(lambda);
    real log_density = 0.0;
    matrix[D,D] Li[K];
    vector[K] loglambda = log(lambda);
    vector[K] theta;
    
    for (k in 1:K){
      Li[k] = tri_inverse(L[k]);
      loglambda[k] += -0.5*D*log(2*pi());
      for (d in 1:D){
        loglambda[k] += log(Li[k][d,d]);
      }
    }
    
    for (n in 1:N){
      for (k in 1:K){
        theta[k] = loglambda[k] - 0.5 * dot_self(Li[k] * (y[n] - mu[k])');
      }
      log_density += log_sum_exp(theta);
    }
    return log_density;
 }
 row_vector col_sums(matrix A){
   int M = rows(A);
   int N = cols(A);
   row_vector[N] column_sums = rep_row_vector(0, N);
   for (m in 1:M){
     column_sums += A[m,:];
   }
   return column_sums;
 }
}
data {
  int N;
  int K;
  int D;
  vector[D] y[N];
  real mu_mu0;
  real mu_sigma0;
  real lSigma_mu0;
  real lSigma_sigma0;
  real eta;
  real PES;
}
transformed data{
  real alpha = PES / K;
  real beta = PES - alpha;
  real neg_log_K = -log(K); // = log(1 / K);
}
parameters {
  vector[D] mu[K];
  vector<lower=0>[D] Sigma[K];
  cholesky_factor_corr[D] LKJ[K];
  // simplex[K] lambda;
}
transformed parameters{
  real log_density = 0.0;
  matrix[D,D] L[K];
  vector[K] theta[N];
  vector[K] lambda = rep_vector(0, K);
  for (k in 1:K){
    L[k] = diag_pre_multiply(Sigma[k], LKJ[k]);
  }
  for (n in 1:N){
    for (k in 1:K){
      theta[n,k] = exp(neg_log_K + multi_normal_cholesky_lpdf(y[n] | mu[k], L[k]));
    }
    lambda += theta[n,:];
    log_density += log(sum(theta[n,:]));
  }
  lambda /= sum(lambda);
  // lambda = column_sums(theta);
}
model {
  target += log_density;
  lambda ~ beta(alpha, beta);
  for (k in 1:K){
    mu[k] ~ normal(mu_mu0, mu_sigma0);
    Sigma[k] ~ lognormal(lSigma_mu0, lSigma_sigma0);
    LKJ[k] ~ lkj_corr_cholesky(eta);
  }
}
