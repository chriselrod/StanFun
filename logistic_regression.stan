data {
  int N;
  int P;
  int<lower=0,upper=1> y;
  matrix[N,P] X;
}
parameters{
  real alpha;
  vector[P] beta;
}
model {
  y ~ bernoulli_logit(alpha + X * beta);
}