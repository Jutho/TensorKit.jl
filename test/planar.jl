using TensorKit, TensorOperations, Test
using TensorKit: planaradd!, planartrace!, planarcontract!
using TensorKit: PlanarTrivial, ℙ

"""
    force_planar(obj)

Replace an object with a planar equivalent -- i.e. one that disallows braiding.
"""
force_planar(V::ComplexSpace) = isdual(V) ? (ℙ^dim(V))' : ℙ^dim(V)
function force_planar(V::GradedSpace)
    return GradedSpace((c ⊠ PlanarTrivial() => dim(V, c) for c in sectors(V))..., isdual(V))
end
force_planar(V::ProductSpace) = mapreduce(force_planar, ⊗, V)
function force_planar(tsrc::TensorMap{<:Any,ComplexSpace})
    tdst = TensorMap{scalartype(tsrc)}(undef,
                                       force_planar(codomain(tsrc)) ←
                                       force_planar(domain(tsrc)))
    copyto!(blocks(tdst)[PlanarTrivial()], blocks(tsrc)[Trivial()])
    return tdst
end
function force_planar(tsrc::TensorMap{<:Any,<:GradedSpace})
    tdst = TensorMap{scalartype(tsrc)}(undef,
                                       force_planar(codomain(tsrc)) ←
                                       force_planar(domain(tsrc)))
    for (c, b) in blocks(tsrc)
        copyto!(blocks(tdst)[c ⊠ PlanarTrivial()], b)
    end
    return tdst
end

@testset "planar methods" verbose = true begin
    @testset "planaradd" begin
        A = randn(ℂ^2 ⊗ ℂ^3 ← ℂ^6 ⊗ ℂ^5 ⊗ ℂ^4)
        C = randn((ℂ^5)' ⊗ (ℂ^6)' ← ℂ^4 ⊗ (ℂ^3)' ⊗ (ℂ^2)')
        A′ = force_planar(A)
        C′ = force_planar(C)
        p = ((4, 3), (5, 2, 1))

        @test force_planar(tensoradd!(C, p, A, :N, true, true)) ≈
              planaradd!(C′, A′, p, true, true)
    end

    @testset "planartrace" begin
        A = randn(ℂ^2 ⊗ ℂ^3 ← ℂ^2 ⊗ ℂ^5 ⊗ ℂ^4)
        C = randn((ℂ^5)' ⊗ ℂ^3 ← ℂ^4)
        A′ = force_planar(A)
        C′ = force_planar(C)
        p = ((4, 2), (5,))
        q = ((1,), (3,))

        @test force_planar(tensortrace!(C, p, A, q, :N, true, true)) ≈
              planartrace!(C′, A′, p, q, true, true)
    end

    @testset "planarcontract" begin
        A = randn(ℂ^2 ⊗ ℂ^3 ← ℂ^2 ⊗ ℂ^5 ⊗ ℂ^4)
        B = randn(ℂ^2 ⊗ ℂ^4 ← ℂ^4 ⊗ ℂ^3)
        C = randn((ℂ^5)' ⊗ (ℂ^2)' ⊗ ℂ^2 ← (ℂ^2)' ⊗ ℂ^4)

        A′ = force_planar(A)
        B′ = force_planar(B)
        C′ = force_planar(C)

        pA = ((1, 3, 4), (5, 2))
        pB = ((2, 4), (1, 3))
        pAB = ((3, 2, 1), (4, 5))

        @test force_planar(tensorcontract!(C, pAB, A, pA, :N, B, pB, :N, true, true)) ≈
              planarcontract!(C′, A′, pA, B′, pB, pAB, true, true)
    end
end

@testset "@planar" verbose = true begin
    T = ComplexF64

    @testset "contractcheck" begin
        V = ℂ^2
        A = rand(T, V ⊗ V ← V)
        B = rand(T, V ⊗ V ← V')
        @tensor C1[i j; k l] := A[i j; m] * B[k l; m]
        @tensor contractcheck = true C2[i j; k l] := A[i j; m] * B[k l; m]
        @test C1 ≈ C2
        B2 = rand(T, V ⊗ V ← V) # wrong duality for third space
        @test_throws SpaceMismatch("incompatible spaces for m: $V ≠ $(V')") begin
            @tensor contractcheck = true C3[i j; k l] := A[i j; m] * B2[k l; m]
        end

        A = rand(T, V ← V ⊗ V)
        B = rand(T, V ⊗ V ← V)
        @planar C1[i; j] := A[i; k l] * τ[k l; m n] * B[m n; j]
        @planar contractcheck = true C2[i; j] := A[i; k l] * τ[k l; m n] * B[m n; j]
        @test C1 ≈ C2
        @test_throws SpaceMismatch("incompatible spaces for m: $V ≠ $(V')") begin
            @planar contractcheck = true C3[i; j] := A[i; k l] * τ[k l; m n] * B[n j; m]
        end
    end

    @testset "MPS networks" begin
        P = ℂ^2
        Vmps = ℂ^12
        Vmpo = ℂ^4

        # ∂AC
        # -------
        x = randn(T, Vmps ⊗ P ← Vmps)
        O = randn(T, Vmpo ⊗ P ← P ⊗ Vmpo)
        GL = randn(T, Vmps ⊗ Vmpo' ← Vmps)
        GR = randn(T, Vmps ⊗ Vmpo ← Vmps)

        x′ = force_planar(x)
        O′ = force_planar(O)
        GL′ = force_planar(GL)
        GR′ = force_planar(GR)

        @tensor y[-1 -2; -3] := GL[-1 2; 1] * x[1 3; 4] * O[2 -2; 3 5] * GR[4 5; -3]
        @planar y′[-1 -2; -3] := GL′[-1 2; 1] * x′[1 3; 4] * O′[2 -2; 3 5] * GR′[4 5; -3]
        @test force_planar(y) ≈ y′

        # ∂AC2
        # -------
        x2 = randn(T, Vmps ⊗ P ← Vmps ⊗ P')
        x2′ = force_planar(x2)
        @tensor contractcheck = true y2[-1 -2; -3 -4] := GL[-1 7; 6] * x2[6 5; 1 3] *
                                                         O[7 -2; 5 4] * O[4 -4; 3 2] *
                                                         GR[1 2; -3]
        @planar y2′[-1 -2; -3 -4] := GL′[-1 7; 6] * x2′[6 5; 1 3] * O′[7 -2; 5 4] *
                                     O′[4 -4; 3 2] * GR′[1 2; -3]
        @test force_planar(y2) ≈ y2′

        # transfer matrix
        # ----------------
        v = randn(T, Vmps ← Vmps)
        v′ = force_planar(v)
        @tensor ρ[-1; -2] := x[-1 2; 1] * conj(x[-2 2; 3]) * v[1; 3]
        @planar ρ′[-1; -2] := x′[-1 2; 1] * conj(x′[-2 2; 3]) * v′[1; 3]
        @test force_planar(ρ) ≈ ρ′

        @tensor ρ2[-1 -2; -3] := GL[1 -2; 3] * x[3 2; -3] * conj(x[1 2; -1])
        @plansor ρ3[-1 -2; -3] := GL[1 2; 4] * x[4 5; -3] * τ[2 3; 5 -2] * conj(x[1 3; -1])
        @planar ρ2′[-1 -2; -3] := GL′[1 2; 4] * x′[4 5; -3] * τ[2 3; 5 -2] *
                                  conj(x′[1 3; -1])
        @test force_planar(ρ2) ≈ ρ2′
        @test ρ2 ≈ ρ3

        # Periodic boundary conditions
        # ----------------------------
        f1 = isomorphism(storagetype(O), fuse(Vmpo^3), Vmpo ⊗ Vmpo' ⊗ Vmpo)
        f2 = isomorphism(storagetype(O), fuse(Vmpo^3), Vmpo ⊗ Vmpo' ⊗ Vmpo)
        f1′ = force_planar(f1)
        f2′ = force_planar(f2)
        @tensor O_periodic1[-1 -2; -3 -4] := O[1 -2; -3 2] * f1[-1; 1 3 4] *
                                             conj(f2[-4; 2 3 4])
        @plansor O_periodic2[-1 -2; -3 -4] := O[1 2; -3 6] * f1[-1; 1 3 5] *
                                              conj(f2[-4; 6 7 8]) * τ[2 3; 7 4] *
                                              τ[4 5; 8 -2]
        @planar O_periodic′[-1 -2; -3 -4] := O′[1 2; -3 6] * f1′[-1; 1 3 5] *
                                             conj(f2′[-4; 6 7 8]) * τ[2 3; 7 4] *
                                             τ[4 5; 8 -2]
        @test O_periodic1 ≈ O_periodic2
        @test force_planar(O_periodic1) ≈ O_periodic′
    end

    @testset "MERA networks" begin
        Vmera = ℂ^2

        u = randn(T, Vmera ⊗ Vmera ← Vmera ⊗ Vmera)
        w = randn(T, Vmera ⊗ Vmera ← Vmera)
        ρ = randn(T, Vmera ⊗ Vmera ⊗ Vmera ← Vmera ⊗ Vmera ⊗ Vmera)
        h = randn(T, Vmera ⊗ Vmera ⊗ Vmera ← Vmera ⊗ Vmera ⊗ Vmera)

        u′ = force_planar(u)
        w′ = force_planar(w)
        ρ′ = force_planar(ρ)
        h′ = force_planar(h)

        @tensor begin
            C = (((((((h[9 3 4; 5 1 2] * u[1 2; 7 12]) * conj(u[3 4; 11 13])) *
                     (u[8 5; 15 6] * w[6 7; 19])) *
                    (conj(u[8 9; 17 10]) * conj(w[10 11; 22]))) *
                   ((w[12 14; 20] * conj(w[13 14; 23])) * ρ[18 19 20; 21 22 23])) *
                  w[16 15; 18]) * conj(w[16 17; 21]))
        end
        @planar begin
            C′ = (((((((h′[9 3 4; 5 1 2] * u′[1 2; 7 12]) * conj(u′[3 4; 11 13])) *
                      (u′[8 5; 15 6] * w′[6 7; 19])) *
                     (conj(u′[8 9; 17 10]) * conj(w′[10 11; 22]))) *
                    ((w′[12 14; 20] * conj(w′[13 14; 23])) * ρ′[18 19 20; 21 22 23])) *
                   w′[16 15; 18]) * conj(w′[16 17; 21]))
        end
        @test C ≈ C′
    end

    @testset "Issue 93" begin
        T = Float64
        V1 = ℂ^2
        V2 = ℂ^3
        t1 = rand(T, V1 ← V2)
        t2 = rand(T, V2 ← V1)

        tr1 = @planar opt = true t1[a; b] * t2[b; a] / 2
        tr2 = @planar opt = true t1[d; a] * t2[b; c] * 1 / 2 * τ[c b; a d]
        tr3 = @planar opt = true t1[d; a] * t2[b; c] * τ[a c; d b] / 2
        tr4 = @planar opt = true t1[f; a] * 1 / 2 * t2[c; d] * τ[d b; c e] * τ[e b; a f]
        tr5 = @planar opt = true t1[f; a] * t2[c; d] / 2 * τ[d b; c e] * τ[a e; f b]
        tr6 = @planar opt = true t1[f; a] * t2[c; d] * τ[c d; e b] / 2 * τ[e b; a f]
        tr7 = @planar opt = true t1[f; a] * t2[c; d] * (τ[c d; e b] * τ[a e; f b] / 2)

        @test tr1 ≈ tr2 ≈ tr3 ≈ tr4 ≈ tr5 ≈ tr6 ≈ tr7

        tr1 = @plansor opt = true t1[a; b] * t2[b; a] / 2
        tr2 = @plansor opt = true t1[d; a] * t2[b; c] * 1 / 2 * τ[c b; a d]
        tr3 = @plansor opt = true t1[d; a] * t2[b; c] * τ[a c; d b] / 2
        tr4 = @plansor opt = true t1[f; a] * 1 / 2 * t2[c; d] * τ[d b; c e] * τ[e b; a f]
        tr5 = @plansor opt = true t1[f; a] * t2[c; d] / 2 * τ[d b; c e] * τ[a e; f b]
        tr6 = @plansor opt = true t1[f; a] * t2[c; d] * τ[c d; e b] / 2 * τ[e b; a f]
        tr7 = @plansor opt = true t1[f; a] * t2[c; d] * (τ[c d; e b] * τ[a e; f b] / 2)

        @test tr1 ≈ tr2 ≈ tr3 ≈ tr4 ≈ tr5 ≈ tr6 ≈ tr7
    end
end
