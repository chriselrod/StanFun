functions{
  real misclassified_dual_binomial_lpmf(int[,] y, vector theta, real[] S, real[] C){
    int K = num_elements(theta);
    vector[K] thetac = 1 - theta;
    real S1c = 1 - S[1];
    real S2c = 1 - S[2];
    real C1c = 1 - C[1];
    real C2c = 1 - C[2];
    
    real S1_S2_ = S[1] * S[2];
    real S1_S2c = S[1] * S2c ;
    real S1cS2_ = S1c  * S[2];
    real S1cS2c = S1c  * S2c ;
    
    real C1_C2_ = C[1] * C[2];
    real C1_C2c = C[1] * C2c ;
    real C1cC2_ = C1c  * C[2];
    real C1cC2c = C1c  * C1c ;

    real log_density = 0.0;
    
    for (k in 1:K){
      log_density += y[k,1] * log( theta[k]*S1_S2_ + thetac[k]*C1cC2c ); // y[1] ++
      log_density += y[k,2] * log( theta[k]*S1_S2c + thetac[k]*C1cC2_ ); // y[2] +-
      log_density += y[k,3] * log( theta[k]*S1cS2_ + thetac[k]*C1_C2c ); // y[3] -+
      log_density += y[k,4] * log( theta[k]*S1cS2c + thetac[k]*C1_C2_ ); // y[4] --
      
      log_density += y[k,5] * log( theta[k]*S[1]   + thetac[k]*C1c    );
      log_density += y[k,6] * log( theta[k]*S1c    + thetac[k]*C[1]   );
      
      log_density += y[k,7] * log( theta[k]*S[2]   + thetac[k]*C2c    );
      log_density += y[k,8] * log( theta[k]*S2c    + thetac[k]*C[2]   );
    }
    
    
    return log_density;
  }
  real misclassified_dual_binomiald_lpdf(matrix y, vector theta, real[] S, real[] C){
    int K = num_elements(theta);
    vector[K] thetac = 1 - theta;
    real S1c = 1 - S[1];
    real S2c = 1 - S[2];
    real C1c = 1 - C[1];
    real C2c = 1 - C[2];
    
    real S1_S2_ = S[1] * S[2];
    real S1_S2c = S[1] * S2c ;
    real S1cS2_ = S1c  * S[2];
    real S1cS2c = S1c  * S2c ;
    
    real C1_C2_ = C[1] * C[2];
    real C1_C2c = C[1] * C2c ;
    real C1cC2_ = C1c  * C[2];
    real C1cC2c = C1c  * C1c ;

    real log_density = 0.0;
    
    for (k in 1:K){
      log_density += y[k,1] * log( theta[k]*S1_S2_ + thetac[k]*C1cC2c ); // y[1] ++
      log_density += y[k,2] * log( theta[k]*S1_S2c + thetac[k]*C1cC2_ ); // y[2] +-
      log_density += y[k,3] * log( theta[k]*S1cS2_ + thetac[k]*C1_C2c ); // y[3] -+
      log_density += y[k,4] * log( theta[k]*S1cS2c + thetac[k]*C1_C2_ ); // y[4] --
      
      log_density += y[k,5] * log( theta[k]*S[1]   + thetac[k]*C1c    );
      log_density += y[k,6] * log( theta[k]*S1c    + thetac[k]*C[1]   );
      
      log_density += y[k,7] * log( theta[k]*S[2]   + thetac[k]*C2c    );
      log_density += y[k,8] * log( theta[k]*S2c    + thetac[k]*C[2]   );
    }
    
    
    return log_density;
  }
}
data{
  int K;
  int y[K,8];
  real alpha_theta;
  real beta_theta;
  real alpha_sensitivity;
  real beta_sensitivity;
  real alpha_specificity;
  real beta_specificity;
}
parameters{
  vector<lower=0,upper=1>[K] theta;
  real<lower=0.5,upper=1> sensitivity[2];
  real<lower=0.5,upper=1> specificity[2];
}
model{
  theta ~ beta(alpha_theta, beta_theta);
  sensitivity ~ beta(alpha_sensitivity, beta_sensitivity);
  specificity ~ beta(alpha_specificity, beta_specificity);
  y ~ misclassified_dual_binomial(theta, sensitivity, specificity);
}
