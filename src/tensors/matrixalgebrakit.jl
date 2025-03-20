# Generic
# -------
for f in (:eig_full, :eig_vals, :eig_trunc, :eigh_full, :eigh_vals, :eigh_trunc, :svd_full,
          :svd_compact, :svd_vals, :svd_trunc)
    @eval function MatrixAlgebraKit.copy_input(::typeof($f),
                                               t::AbstractTensorMap{<:BlasFloat})
        T = factorisation_scalartype($f, t)
        return copy_oftype(t, T)
    end
end

# TODO: move to MatrixAlgebraKit?
macro check_eltype(x, y, f=:identity, g=:eltype)
    msg = "unexpected scalar type: "
    msg *= string(g) * "(" * string(x) * ") != "
    if f == :identity
        msg *= string(g) * "(" * string(y) * ")"
    else
        msg *= string(f) * "(" * string(y) * ")"
    end
    return :($g($x) == $f($g($y)) || throw(ArgumentError($msg)))
end

# function factorisation_scalartype(::typeof(MAK.eig_full!), t::AbstractTensorMap)
#     T = scalartype(t)
#     return promote_type(Float32, typeof(zero(T) / sqrt(abs2(one(T)))))
# end

# Singular value decomposition
# ----------------------------
const _T_USVᴴ = Tuple{<:AbstractTensorMap,<:AbstractTensorMap,<:AbstractTensorMap}
const _T_USVᴴ_diag = Tuple{<:AbstractTensorMap,<:DiagonalTensorMap,<:AbstractTensorMap}

function MatrixAlgebraKit.check_input(::typeof(svd_full!), t::AbstractTensorMap,
                                      (U, S, Vᴴ)::_T_USVᴴ)
    # scalartype checks
    @check_eltype U t
    @check_eltype S t real
    @check_eltype Vᴴ t

    # space checks
    V_cod = fuse(codomain(t))
    V_dom = fuse(domain(t))
    space(U) == (codomain(t) ← V_cod) ||
        throw(SpaceMismatch("`svd_full!(t, (U, S, Vᴴ))` requires `space(U) == (codomain(t) ← fuse(domain(t)))`"))
    space(S) == (V_cod ← V_dom) ||
        throw(SpaceMismatch("`svd_full!(t, (U, S, Vᴴ))` requires `space(S) == (fuse(codomain(t)) ← fuse(domain(t))`"))
    space(Vᴴ) == (V_dom ← domain(t)) ||
        throw(SpaceMismatch("`svd_full!(t, (U, S, Vᴴ))` requires `space(Vᴴ) == (fuse(domain(t)) ← domain(t))`"))

    return nothing
end

function MatrixAlgebraKit.check_input(::typeof(svd_compact!), t::AbstractTensorMap,
                                      (U, S, Vᴴ)::_T_USVᴴ_diag)
    # scalartype checks
    @check_eltype U t
    @check_eltype S t real
    @check_eltype Vᴴ t

    # space checks
    V_cod = V_dom = infimum(fuse(codomain(t)), fuse(domain(t)))
    space(U) == (codomain(t) ← V_cod) ||
        throw(SpaceMismatch("`svd_compact!(t, (U, S, Vᴴ))` requires `space(U) == (codomain(t) ← infimum(fuse(domain(t)), fuse(codomain(t)))`"))
    space(S) == (V_cod ← V_dom) ||
        throw(SpaceMismatch("`svd_compact!(t, (U, S, Vᴴ))` requires diagonal `S` with `domain(S) == (infimum(fuse(codomain(t)), fuse(domain(t)))`"))
    space(Vᴴ) == (V_dom ← domain(t)) ||
        throw(SpaceMismatch("`svd_compact!(t, (U, S, Vᴴ))` requires `space(Vᴴ) == (infimum(fuse(domain(t)), fuse(codomain(t))) ← domain(t))`"))

    return nothing
end

# TODO: svd_vals

function MatrixAlgebraKit.initialize_output(::typeof(svd_full!), t::AbstractTensorMap,
                                            ::MatrixAlgebraKit.AbstractAlgorithm)
    V_cod = fuse(codomain(t))
    V_dom = fuse(domain(t))
    U = similar(t, domain(t) ← V_cod)
    S = similar(t, real(scalartype(t)), V_cod ← V_dom)
    Vᴴ = similar(t, domain(t) ← V_dom)
    return U, S, Vᴴ
end

function MatrixAlgebraKit.initialize_output(::typeof(svd_compact!), t::AbstractTensorMap,
                                            ::MatrixAlgebraKit.AbstractAlgorithm)
    V_cod = V_dom = infimum(fuse(codomain(t)), fuse(domain(t)))
    U = similar(t, domain(t) ← V_cod)
    S = DiagonalTensorMap{real(scalartype(t))}(undef, V_cod ← V_dom)
    Vᴴ = similar(t, domain(t) ← V_dom)
    return U, S, Vᴴ
end

# TODO: svd_vals

function MatrixAlgebraKit.svd_full!(t::AbstractTensorMap, (U, S, Vᴴ),
                                    alg::BlockAlgorithm)
    MatrixAlgebraKit.check_input(svd_full!, t, (U, S, Vᴴ))

    foreachblock(t, U, S, Vᴴ; alg.scheduler) do _, (b, u, s, vᴴ)
        if isempty(b) # TODO: remove once MatrixAlgebraKit supports empty matrices
            one!(length(u) > 0 ? u : vᴴ)
            zerovector!(s)
        else
            u′, s′, vᴴ′ = MatrixAlgebraKit.svd_full!(b, (u, s, vᴴ), alg.alg)
            # deal with the case where the output is not the same as the input
            u === u′ || copyto!(u, u′)
            s === s′ || copyto!(s, s′)
            vᴴ === vᴴ′ || copyto!(vᴴ, vᴴ′)
        end
        return nothing
    end

    return U, S, Vᴴ
end

function MatrixAlgebraKit.svd_compact!(t::AbstractTensorMap, (U, S, Vᴴ),
                                       alg::BlockAlgorithm)
    MatrixAlgebraKit.check_input(svd_compact!, t, (U, S, Vᴴ))

    foreachblock(t, U, S, Vᴴ; alg.scheduler) do _, (b, u, s, vᴴ)
        u′, s′, vᴴ′ = svd_compact!(b, (u, s, vᴴ), alg.alg)
        # deal with the case where the output is not the same as the input
        u === u′ || copyto!(u, u′)
        s === s′ || copyto!(s, s′)
        vᴴ === vᴴ′ || copyto!(vᴴ, vᴴ′)
        return nothing
    end

    return U, S, Vᴴ
end

function MatrixAlgebraKit.default_svd_algorithm(t::AbstractTensorMap{<:BlasFloat};
                                                scheduler=default_blockscheduler(t),
                                                kwargs...)
    return BlockAlgorithm(LAPACK_DivideAndConquer(; kwargs...), scheduler)
end

# Eigenvalue decomposition
# ------------------------
const _T_DV = Tuple{<:DiagonalTensorMap,<:AbstractTensorMap}
function MatrixAlgebraKit.check_input(::typeof(eigh_full!), t::AbstractTensorMap,
                                      (D, V)::_T_DV)
    domain(t) == codomain(t) ||
        throw(ArgumentError("Eigenvalue decomposition requires square input tensor"))

    # scalartype checks
    @check_eltype D t real
    @check_eltype V t

    # space checks
    V_D = fuse(domain(t))
    V_D == space(D, 1) ||
        throw(SpaceMismatch("`eigh_full!(t, (D, V))` requires diagonal `D` with `domain(D) == fuse(domain(t))`"))
    space(V) == (codomain(t) ← V_D) ||
        throw(SpaceMismatch("`eigh_full!(t, (D, V))` requires `space(V) == (codomain(t) ← fuse(domain(t)))`"))

    return nothing
end

function MatrixAlgebraKit.check_input(::typeof(eig_full!), t::AbstractTensorMap,
                                      (D, V)::_T_DV)
    domain(t) == codomain(t) ||
        throw(ArgumentError("Eigenvalue decomposition requires square input tensor"))

    # scalartype checks
    @check_eltype D t complex
    @check_eltype V t complex

    # space checks
    V_D = fuse(domain(t))
    V_D == space(D, 1) ||
        throw(SpaceMismatch("`eig_full!(t, (D, V))` requires diagonal `D` with `domain(D) == fuse(domain(t))`"))
    space(V) == (codomain(t) ← V_D) ||
        throw(SpaceMismatch("`eig_full!(t, (D, V))` requires `space(V) == (codomain(t) ← fuse(domain(t)))`"))

    return nothing
end

function MatrixAlgebraKit.initialize_output(::typeof(eigh_full!), t::AbstractTensorMap,
                                            ::MatrixAlgebraKit.AbstractAlgorithm)
    V_D = fuse(domain(t))
    T = real(scalartype(t))
    D = DiagonalTensorMap{T}(undef, V_D)
    V = similar(t, codomain(t) ← V_D)
    return D, V
end

function MatrixAlgebraKit.initialize_output(::typeof(eig_full!), t::AbstractTensorMap,
                                            ::MatrixAlgebraKit.AbstractAlgorithm)
    V_D = fuse(domain(t))
    Tc = complex(scalartype(t))
    D = DiagonalTensorMap{Tc}(undef, V_D)
    V = similar(t, Tc, codomain(t) ← V_D)
    return D, V
end

for f in (:eigh_full!, :eig_full!)
    @eval function MatrixAlgebraKit.$f(t::AbstractTensorMap, (D, V),
                                       alg::BlockAlgorithm)
        MatrixAlgebraKit.check_input($f, t, (D, V))

        foreachblock(t, D, V; alg.scheduler) do _, (b, d, v)
            d′, v′ = $f(b, (d, v), alg.alg)
            # deal with the case where the output is not the same as the input
            d === d′ || copyto!(d, d′)
            v === v′ || copyto!(v, v′)
            return nothing
        end

        return D, V
    end
end

function MatrixAlgebraKit.default_eig_algorithm(t::AbstractTensorMap{<:BlasFloat};
                                                scheduler=default_blockscheduler(t),
                                                kwargs...)
    return BlockAlgorithm(LAPACK_Expert(; kwargs...), scheduler)
end
function MatrixAlgebraKit.default_eigh_algorithm(t::AbstractTensorMap{<:BlasFloat};
                                                 scheduler=default_blockscheduler(t),
                                                 kwargs...)
    return BlockAlgorithm(LAPACK_MultipleRelativelyRobustRepresentations(; kwargs...),
                          scheduler)
end
