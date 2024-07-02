function ChainRulesCore.rrule(::typeof(TO.tensorcontract!),
                              C::AbstractTensorMap{S}, pC::Index2Tuple,
                              A::AbstractTensorMap{S}, pA::Index2Tuple, conjA::Symbol,
                              B::AbstractTensorMap{S}, pB::Index2Tuple, conjB::Symbol,
                              α::Number, β::Number,
                              backend::Backend...) where {S}
    C′ = tensorcontract!(copy(C), pC, A, pA, conjA, B, pB, conjB, α, β, backend...)

    projectA = ProjectTo(A)
    projectB = ProjectTo(B)
    projectC = ProjectTo(C)
    projectα = ProjectTo(α)
    projectβ = ProjectTo(β)

    function pullback(ΔC′)
        ΔC = unthunk(ΔC′)
        ipC = invperm(linearize(pC))
        pΔC = (TupleTools.getindices(ipC, trivtuple(TO.numout(pA))),
               TupleTools.getindices(ipC, TO.numout(pA) .+ trivtuple(TO.numin(pB))))

        dC = @thunk projectC(scale(ΔC, conj(β)))
        dA = @thunk begin
            ipA = (invperm(linearize(pA)), ())
            conjΔC = conjA == :C ? :C : :N
            conjB′ = conjA == :C ? conjB : _conj(conjB)
            _dA = zerovector(A,
                             promote_contract(scalartype(ΔC), scalartype(B), scalartype(α)))
            tB = twist(B,
                       TupleTools.vcat(filter(x -> !isdual(space(B, x)), pB[1]),
                                       filter(x -> isdual(space(B, x)), pB[2])))
            _dA = tensorcontract!(_dA, ipA,
                                  ΔC, pΔC, conjΔC,
                                  tB, reverse(pB), conjB′,
                                  conjA == :C ? α : conj(α), Zero(), backend...)
            return projectA(_dA)
        end
        dB = @thunk begin
            ipB = (invperm(linearize(pB)), ())
            conjΔC = conjB == :C ? :C : :N
            conjA′ = conjB == :C ? conjA : _conj(conjA)
            _dB = zerovector(B,
                             promote_contract(scalartype(ΔC), scalartype(A), scalartype(α)))
            tA = twist(A,
                       TupleTools.vcat(filter(x -> isdual(space(A, x)), pA[1]),
                                       filter(x -> !isdual(space(A, x)), pA[2])))
            _dB = tensorcontract!(_dB, ipB,
                                  tA, reverse(pA), conjA′,
                                  ΔC, pΔC, conjΔC,
                                  conjB == :C ? α : conj(α), Zero(), backend...)
            return projectB(_dB)
        end
        dα = @thunk begin
            # TODO: this result should be AB = (C′ - βC) / α as C′ = βC + αAB
            AB = tensorcontract(pC, A, pA, conjA, B, pB, conjB)
            return projectα(inner(AB, ΔC))
        end
        dβ = @thunk projectβ(inner(C, ΔC))
        dbackend = map(x -> NoTangent(), backend)
        return NoTangent(), dC, NoTangent(),
               dA, NoTangent(), NoTangent(), dB, NoTangent(), NoTangent(), dα, dβ,
               dbackend...
    end
    return C′, pullback
end

function ChainRulesCore.rrule(::typeof(TO.tensoradd!),
                              C::AbstractTensorMap{S}, pC::Index2Tuple,
                              A::AbstractTensorMap{S}, conjA::Symbol,
                              α::Number, β::Number, backend::Backend...) where {S}
    C′ = tensoradd!(copy(C), pC, A, conjA, α, β, backend...)

    projectA = ProjectTo(A)
    projectC = ProjectTo(C)
    projectα = ProjectTo(α)
    projectβ = ProjectTo(β)

    function pullback(ΔC′)
        ΔC = unthunk(ΔC′)
        dC = @thunk projectC(scale(ΔC, conj(β)))
        dA = @thunk begin
            ipC = invperm(linearize(pC))
            _dA = zerovector(A, promote_add(ΔC, α))
            _dA = tensoradd!(_dA, (ipC, ()), ΔC, conjA, conjA == :N ? conj(α) : α, Zero(),
                             backend...)
            return projectA(_dA)
        end
        dα = @thunk begin
            # TODO: this is an inner product implemented as a contraction
            # for non-symmetric tensors this might be more efficient like this,
            # but for symmetric tensors an intermediate object will anyways be created
            # and then it might be more efficient to use an addition and inner product
            tΔC = twist(ΔC, filter(x -> isdual(space(ΔC, x)), allind(ΔC)))
            _dα = tensorscalar(tensorcontract(((), ()), A, ((), linearize(pC)),
                                              _conj(conjA), tΔC,
                                              (trivtuple(TO.numind(pC)),
                                               ()), :N, One(), backend...))
            return projectα(_dα)
        end
        dβ = @thunk projectβ(inner(C, ΔC))
        dbackend = map(x -> NoTangent(), backend)
        return NoTangent(), dC, NoTangent(), dA, NoTangent(), dα, dβ, dbackend...
    end

    return C′, pullback
end

function ChainRulesCore.rrule(::typeof(tensortrace!), C::AbstractTensorMap{S},
                              pC::Index2Tuple, A::AbstractTensorMap{S},
                              pA::Index2Tuple, conjA::Symbol, α::Number, β::Number,
                              backend::Backend...) where {S}
    C′ = tensortrace!(copy(C), pC, A, pA, conjA, α, β, backend...)

    projectA = ProjectTo(A)
    projectC = ProjectTo(C)
    projectα = ProjectTo(α)
    projectβ = ProjectTo(β)

    function pullback(ΔC′)
        ΔC = unthunk(ΔC′)
        dC = @thunk projectC(scale(ΔC, conj(β)))
        dA = @thunk begin
            ipC = invperm((linearize(pC)..., pA[1]..., pA[2]...))
            E = one!(TO.tensoralloc_add(scalartype(A), pA, A, conjA))
            twist!(E, filter(x -> !isdual(space(E, x)), codomainind(E)))
            _dA = zerovector(A, promote_scale(ΔC, α))
            _dA = tensorproduct!(_dA, (ipC, ()), ΔC,
                                 (trivtuple(TO.numind(pC)), ()), conjA, E,
                                 ((), trivtuple(TO.numind(pA))), conjA,
                                 conjA == :N ? conj(α) : α, Zero(), backend...)
            return projectA(_dA)
        end
        dα = @thunk begin
            # TODO: this result might be easier to compute as:
            # C′ = βC + α * trace(A) ⟹ At = (C′ - βC) / α
            At = tensortrace(pC, A, pA, conjA)
            return projectα(inner(At, ΔC))
        end
        dβ = @thunk projectβ(inner(C, ΔC))
        dbackend = map(x -> NoTangent(), backend)
        return NoTangent(), dC, NoTangent(), dA, NoTangent(), NoTangent(), dα, dβ,
               dbackend...
    end

    return C′, pullback
end
