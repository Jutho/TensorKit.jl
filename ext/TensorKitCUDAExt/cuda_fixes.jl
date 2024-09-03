# ambiguity fix
function LinearAlgebra.dot(x::Adjoint{T1,<:CuArray{T1,N}},
                           y::Adjoint{T2,<:CuArray{T2,N}}) where {T1<:Union{Real,Complex},
                                                                  T2<:Union{Real,Complex},N}
    return @invoke LinearAlgebra.dot(x::Adjoint{<:Union{Real,Complex}},
                                     y::Adjoint{<:Union{Real,Complex}})
end

# somehow this is not defined for CuArray
function Base.promote_rule(a::Type{CuArray{T₁,N,M}},
                           b::Type{CuArray{T₂,N,M}}) where {T₁,T₂,N,M}
    return Base.el_same(promote_type(T₁, T₂), a, b)
end

function LinearAlgebra._diagm(size, kv::Pair{<:Integer,<:CuVector}...)
    A = LinearAlgebra.diagm_container(size, kv...)
    for p in kv
        inds = LinearAlgebra.diagind(A, p.first)
        A[inds] .+= p.second
    end
    return A
end
function LinearAlgebra.diagm_container(size, kv::Pair{<:Integer,<:CuVector}...)
    T = promote_type(map(x -> eltype(x.second), kv)...)
    # For some type `T`, `zero(T)` is not a `T` and `zeros(T, ...)` fails.
    U = promote_type(T, typeof(zero(T)))
    return CUDA.zeros(U, LinearAlgebra.diagm_size(size, kv...)...)
end
