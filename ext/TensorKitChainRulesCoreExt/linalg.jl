# Linear Algebra chainrules
# -------------------------
function ChainRulesCore.rrule(::typeof(+), a::AbstractTensorMap, b::AbstractTensorMap)
    plus_pullback(Δc) = NoTangent(), Δc, Δc
    return a + b, plus_pullback
end

ChainRulesCore.rrule(::typeof(-), a::AbstractTensorMap) = -a, Δc -> (NoTangent(), -Δc)
function ChainRulesCore.rrule(::typeof(-), a::AbstractTensorMap, b::AbstractTensorMap)
    minus_pullback(Δc) = NoTangent(), Δc, -Δc
    return a - b, minus_pullback
end

function ChainRulesCore.rrule(::typeof(*), a::AbstractTensorMap, b::AbstractTensorMap)
    times_pullback(Δc) = NoTangent(), @thunk(Δc * b'), @thunk(a' * Δc)
    return a * b, times_pullback
end

function ChainRulesCore.rrule(::typeof(*), a::AbstractTensorMap, b::Number)
    times_pullback(Δc) = NoTangent(), @thunk(Δc * b'), @thunk(dot(a, Δc))
    return a * b, times_pullback
end

function ChainRulesCore.rrule(::typeof(*), a::Number, b::AbstractTensorMap)
    times_pullback(Δc) = NoTangent(), @thunk(dot(b, Δc)), @thunk(a' * Δc)
    return a * b, times_pullback
end

function ChainRulesCore.rrule(::typeof(⊗), A::AbstractTensorMap, B::AbstractTensorMap)
    C = A ⊗ B
    projectA = ProjectTo(A)
    projectB = ProjectTo(B)
    function otimes_pullback(ΔC_)
        # TODO: this rule is probably better written in terms of inner products,
        # using planarcontract and adjoint tensormaps would remove the twists.
        ΔC = unthunk(ΔC_)
        pΔC = ((codomainind(A)..., (domainind(A) .+ numout(B))...),
               ((codomainind(B) .+ numout(A))...,
                (domainind(B) .+ (numin(A) + numout(A)))...))
        dA_ = @thunk let
            ipA = (codomainind(A), domainind(A))
            pB = (allind(B), ())
            dA = zerovector(A, promote_contract(scalartype(ΔC), scalartype(B)))
            tB = twist(B, filter(x -> isdual(space(B, x)), allind(B)))
            dA = tensorcontract!(dA, ΔC, pΔC, false, tB, pB, true, ipA)
            return projectA(dA)
        end
        dB_ = @thunk let
            ipB = (codomainind(B), domainind(B))
            pA = ((), allind(A))
            dB = zerovector(B, promote_contract(scalartype(ΔC), scalartype(A)))
            tA = twist(A, filter(x -> isdual(space(A, x)), allind(A)))
            dB = tensorcontract!(dB, tA, pA, true, ΔC, pΔC, false, ipB)
            return projectB(dB)
        end
        return NoTangent(), dA_, dB_
    end
    return C, otimes_pullback
end

function ChainRulesCore.rrule(::typeof(permute), tsrc::AbstractTensorMap, p::Index2Tuple;
                              copy::Bool=false)
    function permute_pullback(Δtdst)
        invp = TensorKit._canonicalize(TupleTools.invperm(linearize(p)), tsrc)
        return NoTangent(), permute(unthunk(Δtdst), invp; copy=true), NoTangent()
    end
    return permute(tsrc, p; copy=true), permute_pullback
end

function ChainRulesCore.rrule(::typeof(tr), A::AbstractTensorMap)
    tr_pullback(Δtr) = NoTangent(), Δtr * id(domain(A))
    return tr(A), tr_pullback
end

function ChainRulesCore.rrule(::typeof(adjoint), A::AbstractTensorMap)
    adjoint_pullback(Δadjoint) = NoTangent(), adjoint(unthunk(Δadjoint))
    return adjoint(A), adjoint_pullback
end

function ChainRulesCore.rrule(::typeof(dot), a::AbstractTensorMap, b::AbstractTensorMap)
    dot_pullback(Δd) = NoTangent(), @thunk(b * Δd'), @thunk(a * Δd)
    return dot(a, b), dot_pullback
end

function ChainRulesCore.rrule(::typeof(norm), a::AbstractTensorMap, p::Real=2)
    p == 2 || error("currently only implemented for p = 2")
    n = norm(a, p)
    function norm_pullback(Δn)
        return NoTangent(), a * (Δn' + Δn) / 2 / hypot(n, eps(one(n))), NoTangent()
    end
    return n, norm_pullback
end

function ChainRulesCore.rrule(::typeof(real), a::AbstractTensorMap)
    a_real = real(a)
    real_pullback(Δa) = NoTangent(), eltype(a) <: Real ? Δa : complex(unthunk(Δa))
    return a_real, real_pullback
end

function ChainRulesCore.rrule(::typeof(imag), a::AbstractTensorMap)
    a_imag = imag(a)
    function imag_pullback(Δa)
        Δa′ = unthunk(Δa)
        return NoTangent(),
               eltype(a) <: Real ? ZeroTangent() : complex(zerovector(Δa′), Δa′)
    end
    return a_imag, imag_pullback
end
