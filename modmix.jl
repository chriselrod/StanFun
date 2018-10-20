

using Memoize
@memoize function fibber(n)
   n < 2 && return n
   fibber(n-1) + fibber(n-2)
end

function inv_triangle!(Ti, T)
    N = size(T,1)
    @inbounds for i ∈ 1:N
      Ti[i,i] = 1/T[i,i]
    end
    @inbounds for j ∈ 1:N-1 # row_displacement
      for i ∈ 1:N-j # column
        x = Ti[j+i,i+1] * T[i+1,i]
        for k ∈ i+2:j+i
          x = x + Ti[j+i,k] * T[k,i]
        end
        Ti[j+i,i] = -Ti[i,i] * x
      end
    end
    Ti
end
# real(8),  parameter                         :: nhlogtau = -9.189385332046727417803297364056D-01 ! -log(2*pi)/2
# integer,                  intent(in)        :: N
# real(8),  dimension(N,N), intent(out)       :: Ti
# real(8),                  intent(out)       :: logdet
# real(8),  dimension(N,N), intent(in)        :: T
# integer                                     :: i, j, k
# real(8)                                     :: x, y
#
# logdet = N * nhlogtau ! only because this is for calculating multivariate normal density. Otherwise 0d0.
# do i = 1,N
#   do j = 1,i-1
#     Ti(j,i) = 0
#   end do
#   Ti(i,i) = 1/T(i,i)
#   logdet = logdet + dlog(Ti(i,i))
# end do
# do j = 1,N-1 ! row_displacement
#   do i = 1,N-j ! column
#     x = Ti(j+i,i+1) * T(i+1,i)
#     do k = i+2,j+i
#       x = x + Ti(j+i,k) * T(k,i)
#     end do
#     Ti(j+i,i) = -Ti(i,i) * x
#   end do
# end do


using StaticArrays
sym(s, i, j) = Symbol(s, :_, i, :_, j)
@generated function invert_ltriangle(L::SMatrix{N,N,T}) where {N,T}
    q = quote
        ld = $(-0.5N*log(2π))
    end
    for n1 ∈ 1:N
        push!(q.args, :($(sym(:Li, n1, n1)) = 1/L[$n1,$n1]))
        push!(q.args, :(ld += log($(sym(:Li, n1, n1)))))
    end
    for n2 ∈ 1:N-1
        for n1 ∈ 1:N-n2
            push!(q.args, :($(sym(:Li, n2+n1,n1)) = $(sym(:Li, n2+n1,n1+1))*L[$(n1+1),$n1] ))
            for n3 ∈ n1+2:n2+n1
                push!(q.args, :($(sym(:Li, n2+n1,n1)) += $(sym(:Li, n2+n1,n3))*L[$n3,$n1] ))

            end
            push!(q.args, :($(sym(:Li, n2+n1,n1)) *= -$(sym(:Li, n1,n1)) ))
        end
    end
    out = :(SMatrix{$N,$N,$T}())
    for n1 ∈ 1:N
        for n2 ∈ 1:n1-1
            push!(out.args, zero(T))
        end
        push!(out.args, sym(:Li, n1, n1))
        for n2 ∈ n1+1:N
            push!(out.args, sym(:Li, n2, n1))
        end
    end
    quote
        @inbounds begin
            $q
        end
        ld, $out
    end
end


@generated function invert_triangle(L::SMatrix{N,N,T}) where {N,T}
    q = quote
        $(Expr(:meta, :inline))
        # ld = $(-0.5N*log(2pi))
    end
    for n1 ∈ 1:N
        push!(q.args, :($(sym(:Li, n1, n1)) = 1/L[$n1,$n1]))
        # push!(q.args, :(ld += log($(sym(:Li, n1, n1)))))
    end
    for n2 ∈ 1:N-1
        for n1 ∈ 1:N-n2
            push!(q.args, :($(sym(:Li, n2+n1,n1)) = $(sym(:Li, n2+n1,n1+1))*L[$(n1+1),$n1] ))
            for n3 ∈ n1+2:n2+n1
                push!(q.args, :($(sym(:Li, n2+n1,n1)) += $(sym(:Li, n2+n1,n3))*L[$n3,$n1] ))

            end
            push!(q.args, :($(sym(:Li, n2+n1,n1)) *= -$(sym(:Li, n1,n1)) ))
        end
    end
    out = :(SMatrix{$N,$N,$T}())
    for n1 ∈ 1:N
        for n2 ∈ 1:n1-1
            push!(out.args, zero(T))
        end
        push!(out.args, sym(:Li, n1, n1))
        for n2 ∈ n1+1:N
            push!(out.args, sym(:Li, n2, n1))
        end
    end
    quote
        @inbounds begin
            $q
        end
        # ld, $out
        $out
    end
end

@inline function mvnormal_lpdf(y, mu, Li, logdet)
    delta = Li * (y - mu)
    logdet - 0.5 * (delta' * delta)
end

@generated function normal_mixture_lpdf(y::Vector{SVector{D,T}}, λ::SVector{K,T}, μ::SMatrix{D,K,T}, L::SArray{Tuple{D,D,K},T}) where {D,K,T}
    quote
        ld = zero($T)
        @inbounds begin
            @nexprs $K k -> begin
                Li_k = invert_triangle(L[:,:,k])
                logdet_k = +($(-0.5D*log(2π)), $([:(log(Li_k[$d,$d])) for d ∈ 1:D]...))
            end
            logλ = log.(λ)
            for n ∈ eachindex(y)
                theta = $(SVector{K,T})((@ntuple $K k -> mvnormal_lpdf(y[n], μ[:,k], Li_k, logdet_k)))
                ld += log(sum(exp.(logλ + theta)))
            end
        end # inbounds
        ld
    end # quote
end

tri(x,::Val{D}) where D = SMatrix{D,D}(cholesky(x' * x).L)

y = [(@SVector randn(4)) for n ∈ 1:150]
λ = @SVector rand(3)
λ /= sum(λ)
μ = @SMatrix randn(4,3)
LA = SArray{Tuple{4,4,3}}(
    tri(randn(6,4),Val(4))...,
    tri(randn(6,4),Val(4))...,
    tri(randn(6,4),Val(4))...
)

normal_mixture_lpdf(y, λ, μ, LA)
