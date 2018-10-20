
module mixture

implicit none


contains


  subroutine invert_triangle(Ti, logdet, T, N)
    real(8),  parameter                         :: nhlogtau = -9.189385332046727417803297364056D-01 ! -log(2*pi)/2
    integer,                  intent(in)        :: N
    real(8),  dimension(N,N), intent(out)       :: Ti
    real(8),                  intent(out)       :: logdet
    real(8),  dimension(N,N), intent(in)        :: T
    integer                                     :: i, j, k
    real(8)                                     :: x, y

    logdet = N * nhlogtau ! only because this is for calculating multivariate normal density. Otherwise 0d0.
    do i = 1,N
      do j = 1,i-1
        Ti(j,i) = 0
      end do
      Ti(i,i) = 1/T(i,i)
      logdet = logdet + dlog(Ti(i,i))
    end do
    do j = 1,N-1 ! row_displacement
      do i = 1,N-j ! column
        x = Ti(j+i,i+1) * T(i+1,i)
        do k = i+2,j+i
          x = x + Ti(j+i,k) * T(k,i)
        end do
        Ti(j+i,i) = -Ti(i,i) * x
      end do
    end do

  end subroutine

  subroutine mvnormal_mixture4(ld, y, lambda, mu, L, N, K) bind(C, name = "MVNormalMixture4")
    real(8),                      intent(out)   ::  ld
    integer,                      intent(in)    ::  N, K
    real(8),  dimension(4, N),    intent(in)    ::  y
    real(8),  dimension(K),       intent(in)    ::  lambda
    real(8),  dimension(4, K),    intent(in)    ::  mu
    real(8),  dimension(4,4, K),  intent(inout) ::  L

    call mvnormal_mixture(ld, y, lambda, mu, L, N, 4, K)

  end subroutine mvnormal_mixture4

  subroutine mvnormal_mixture(ld, y, lambda, mu, L, N, D, K) bind(C, name = "MVNormalMixture")
    real(8),                      intent(out)   ::  ld
    integer,                      intent(in)    ::  N, D, K
    real(8),  dimension(D, N),    intent(in)    ::  y
    real(8),  dimension(K),       intent(in)    ::  lambda
    real(8),  dimension(D, K),    intent(in)    ::  mu
    real(8),  dimension(D,D, K),  intent(inout) ::  L

    real(8),  dimension(K)                      ::  theta, loglambda, LogDets
    real(8),  dimension(D, D, K)                ::  Li
    integer                                     ::  i, j

    ld = 0d0

    do j = 1,K
      call invert_triangle(Li(:,:,j), LogDets(j), L(:,:,j), D)
    end do
    loglambda = log(lambda) + LogDets

    do i = 1,N
      do j = 1,K
        theta(j) = mvnormal_precision_kernel_lpdf(y(:,i), mu(:,j), Li(:,:,j), D)
      end do
      ld = ld + log(sum(exp(loglambda + theta)))
    end do

  end subroutine mvnormal_mixture


  function mvnormal_precision_kernel_lpdf(y, mu, Li, D) result(p)
    integer,                    intent(in)  ::  D
    real(8),  dimension(D),     intent(in)  ::  y, mu
    real(8),  dimension(D, D),  intent(in)  ::  Li
    real(8)                                 ::  p
    real(8),  dimension(D)                  ::  delta

    delta = matmul(Li, y - mu)
    p = - 0.5 * sum(delta*delta)

  end function mvnormal_precision_kernel_lpdf

end module mixture
