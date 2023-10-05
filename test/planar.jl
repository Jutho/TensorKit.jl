println("------------------------------------")
println("Planar")
println("------------------------------------")

using TensorKit: planaradd!, planartrace!, planarcontract!, BraidingTensor,
                 SymmetricBraiding
using TensorOperations

@testset "$(TensorKit.type_repr(I))" verbose = true for I in sectorlist
    V = smallspace(I)
    if isnothing(V)
        "No spaces defined for $(TensorKit.type_repr(I)), skipping tests"
        continue
    end
    Istr = TensorKit.type_repr(I)
    println("Starting tests for $Istr...")
    V1, V2, V3, V4, V5 = V

    if BraidingStyle(I) isa SymmetricBraiding
        @testset "planaradd" begin
            A = TensorMap(randn, V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5)
            C = TensorMap(randn, V4' ⊗ V3' ← V5 ⊗ V2' ⊗ V1')
            p = ((4, 3), (5, 2, 1))

            @test force_planar(tensoradd!(C, p, A, :N, true, true)) ≈
                  planaradd!(force_planar(C), force_planar(A), p, true, true)
        end
        @testset "planartrace" begin
            A = TensorMap(randn, V1 ⊗ V2 ← V1 ⊗ V4 ⊗ V5)
            C = TensorMap(randn, V4' ⊗ V2 ← V5)

            p = ((4, 2), (5,))
            q = ((1,), (3,))

            @test force_planar(tensortrace!(C, p, A, q, :N, true, true)) ≈
                  planartrace!(force_planar(C), force_planar(A), p, q, true, true)
        end

        @testset "planarcontract" begin
            A = TensorMap(randn, V1 ⊗ V2 ← V3 ⊗ V4 ⊗ V5)
            B = TensorMap(randn, V2 ⊗ V5 ← V1 ⊗ V2)
            C = TensorMap(randn, V4' ⊗ V3' ⊗ V1 ← V2' ⊗ V1)

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

    @testset "BraidingTensor conversion" begin
        for (V1, V2) in [(V1, V1), (V1', V1), (V1, V1'), (V1', V1')]
            τ = BraidingTensor(V1, V2)

            @test domain(τ) == V1 ⊗ V2
            @test codomain(τ) == V2 ⊗ V1

            for (c, b) in blocks(copy(τ))
                @test b ≈ block(τ, c)
            end

            @test domain(τ') == codomain(τ)
            @test codomain(τ') == domain(τ)

            for (c, b) in blocks(copy(τ'))
                @test b ≈ block(τ', c)
            end
        end
    end

    t = TensorMap(randn, V1 * V1' * V1' * V1, V1 * V1')

    ττ = copy(BraidingTensor(V1, V1'))
    @planar t1[-1 -2 -3 -4; -5 -6] := τ[-1 -2; 1 2] * t[1 2 -3 -4; -5 -6]
    @planar t2[-1 -2 -3 -4; -5 -6] := ττ[-1 -2; 1 2] * t[1 2 -3 -4; -5 -6]
    @planar t3[-1 -2 -3 -4; -5 -6] := τ[2 1; -2 -1] * t[1 2 -3 -4; -5 -6]
    @planar t4[-1 -2 -3 -4; -5 -6] := τ'[-2 2; -1 1] * t[1 2 -3 -4; -5 -6]
    t5 = braid(t, ((2, 1, 3, 4), (5, 6)), (1, 2, 3, 4, 5, 6))
    @test t1 ≈ t2
    @test t1 ≈ t3
    @test t1 ≈ t4
    @test t1 ≈ t5

    ττ = copy(BraidingTensor(V1', V1'))
    @planar t1[-1 -2 -3 -4; -5 -6] := τ[-2 -3; 1 2] * t[-1 1 2 -4; -5 -6]
    @planar t2[-1 -2 -3 -4; -5 -6] := ττ[-2 -3; 1 2] * t[-1 1 2 -4; -5 -6]
    @planar t3[-1 -2 -3 -4; -5 -6] := τ[2 1; -3 -2] * t[-1 1 2 -4; -5 -6]
    @planar t4[-1 -2 -3 -4; -5 -6] := τ'[-3 2; -2 1] * t[-1 1 2 -4; -5 -6]
    t5 = braid(t, ((1, 3, 2, 4), (5, 6)), (1, 2, 3, 4, 5, 6))
    @test t1 ≈ t2
    @test t1 ≈ t3
    @test t1 ≈ t4
    @test t1 ≈ t5

    ττ = copy(BraidingTensor(V1', V1))
    @planar t1[-1 -2 -3 -4; -5 -6] := τ[1 -2; 2 -3] * t[-1 1 2 -4; -5 -6]
    @planar t2[-1 -2 -3 -4; -5 -6] := ττ[1 -2; 2 -3] * t[-1 1 2 -4; -5 -6]
    @planar t3[-1 -2 -3 -4; -5 -6] := τ[-3 2; -2 1] * t[-1 1 2 -4; -5 -6]
    @planar t4[-1 -2 -3 -4; -5 -6] := τ'[-2 -3; 1 2] * t[-1 1 2 -4; -5 -6]
    # @test t1 ≈ t2
    @test t1 ≈ t3
    @test t1 ≈ t4

    ττ = copy(BraidingTensor(V1, V1'))
    @planar t1[-1 -2 -3 -4; -5 -6] := τ[-3 2; -2 1] * t[-1 1 2 -4; -5 -6]
    @planar t2[-1 -2 -3 -4; -5 -6] := ττ[-3 2; -2 1] * t[-1 1 2 -4; -5 -6]
    @planar t3[-1 -2 -3 -4; -5 -6] := τ[1 -2; 2 -3] * t[-1 1 2 -4; -5 -6]
    @planar t4[-1 -2 -3 -4; -5 -6] := τ'[2 1; -3 -2] * t[-1 1 2 -4; -5 -6]
    # @test t1 ≈ t2
    @test t1 ≈ t3
    @test t1 ≈ t4

    ττ = copy(BraidingTensor(V1, V1))
    @planar t1[-1 -2 -3 -4; -5 -6] := τ[2 1; -3 -2] * t[-1 1 2 -4; -5 -6]
    @planar t2[-1 -2 -3 -4; -5 -6] := ττ[2 1; -3 -2] * t[-1 1 2 -4; -5 -6]
    @planar t3[-1 -2 -3 -4; -5 -6] := τ[-2 -3; 1 2] * t[-1 1 2 -4; -5 -6]
    @planar t4[-1 -2 -3 -4; -5 -6] := τ'[1 -2; 2 -3] * t[-1 1 2 -4; -5 -6]
    # @test t1 ≈ t2
    @test t1 ≈ t3
    @test t1 ≈ t4

    ττ = copy(BraidingTensor(V1', V1))
    @planar t1[-1 -2 -3 -4; -5 -6] := τ[-3 -4; 1 2] * t[-1 -2 1 2; -5 -6]
    @planar t2[-1 -2 -3 -4; -5 -6] := ττ[-3 -4; 1 2] * t[-1 -2 1 2; -5 -6]
    @planar t3[-1 -2 -3 -4; -5 -6] := τ[2 1; -4 -3] * t[-1 -2 1 2; -5 -6]
    @planar t4[-1 -2 -3 -4; -5 -6] := τ'[-4 2; -3 1] * t[-1 -2 1 2; -5 -6]
    # @test t1 ≈ t2
    @test t1 ≈ t3
    @test t1 ≈ t4

    ττ = copy(BraidingTensor(V1', V1))
    @planar t1[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * τ[1 2; -5 -6]
    @planar t2[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * ττ[1 2; -5 -6]
    @planar t3[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * τ[-6 -5; 2 1]
    @planar t4[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * τ'[2 -6; 1 -5]
    @test t1 ≈ t2
    @test t1 ≈ t3
    @test t1 ≈ t4

    ττ = copy(BraidingTensor(V1, V1'))
    @planar t1[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * τ'[1 2; -5 -6]
    @planar t2[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * ττ'[1 2; -5 -6]
    @planar t3[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * τ'[-6 -5; 2 1]
    @planar t4[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 -4; 1 2] * τ[2 -6; 1 -5]
    @test t1 ≈ t2
    @test t1 ≈ t3
    @test t1 ≈ t4

    ττ = copy(BraidingTensor(V1, V1))
    @planar t1[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 1; -5 2] * τ[-4 -6; 1 2]
    @planar t2[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 1; -5 2] * ττ[-4 -6; 1 2]
    @planar t3[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 1; -5 2] * τ[2 1; -6 -4]
    @planar t4[-1 -2 -3 -4; -5 -6] := t[-1 -2 -3 1; -5 2] * τ'[-6 2; -4 1]
    # @test t1 ≈ t2
    @test t1 ≈ t3
    @test t1 ≈ t4

    ττ = copy(BraidingTensor(V1', V1))
    @planar t1[(); (-1, -2)] := τ[2 1; 3 4] * t[1 2 3 4; -1 -2]
    @planar t2[(); (-1, -2)] := ττ[2 1; 3 4] * t[1 2 3 4; -1 -2]
    @planar t3[(); (-1, -2)] := τ[4 3; 1 2] * t[1 2 3 4; -1 -2]
    @planar t4[(); (-1, -2)] := τ'[1 4; 2 3] * t[1 2 3 4; -1 -2]
    @test t1 ≈ t2
    @test t1 ≈ t3
    @test t1 ≈ t4

    ττ = copy(BraidingTensor(V1, V1))
    @planar t1[-1; -2] := τ[2 1; 3 4] * t[-1 1 2 3; -2 4]
    @planar t2[-1; -2] := ττ[2 1; 3 4] * t[-1 1 2 3; -2 4]
    @planar t3[-1; -2] := τ[4 3; 1 2] * t[-1 1 2 3; -2 4]
    @planar t4[-1; -2] := τ'[1 4; 2 3] * t[-1 1 2 3; -2 4]
    @test t1 ≈ t2
    @test t1 ≈ t3
    @test t1 ≈ t4

    ττ = copy(BraidingTensor(V1, V1'))
    @planar t1[-1 -2] := τ[2 1; 3 4] * t[-1 -2 1 2; 4 3]
    @planar t2[-1 -2] := ττ[2 1; 3 4] * t[-1 -2 1 2; 4 3]
    @planar t3[-1 -2] := τ[4 3; 1 2] * t[-1 -2 1 2; 4 3]
    @planar t4[-1 -2] := τ'[1 4; 2 3] * t[-1 -2 1 2; 4 3]
    @test t1 ≈ t2
    @test t1 ≈ t3
    @test t1 ≈ t4

    ττ = copy(BraidingTensor(V1, V1'))
    @planar t1[-1 -2; -3 -4] := τ[-1 3; 1 2] * t[1 2 3 -2; -3 -4]
    @planar t2[-1 -2; -3 -4] := ττ[-1 3; 1 2] * t[1 2 3 -2; -3 -4]
    @planar t3[-1 -2; -3 -4] := τ[2 1; 3 -1] * t[1 2 3 -2; -3 -4]
    @planar t4[-1 -2; -3 -4] := τ'[3 2; -1 1] * t[1 2 3 -2; -3 -4]
    @test t1 ≈ t2
    @test t1 ≈ t3
    @test t1 ≈ t4

    ττ = copy(BraidingTensor(V1', V1'))
    @planar t1[-1 -2; -3 -4] := τ'[-2 3; 1 2] * t[-1 1 2 3; -3 -4]
    @planar t2[-1 -2; -3 -4] := ττ'[-2 3; 1 2] * t[-1 1 2 3; -3 -4]
    @planar t3[-1 -2; -3 -4] := τ'[2 1; 3 -2] * t[-1 1 2 3; -3 -4]
    @planar t4[-1 -2; -3 -4] := τ[3 2; -2 1] * t[-1 1 2 3; -3 -4]
    @test t1 ≈ t2
    @test t1 ≈ t3
    @test t1 ≈ t4

    ττ = copy(BraidingTensor(V1', V1))
    @planar t1[-1 -2 -3; -4] := τ[-3 3; 1 2] * t[-1 -2 1 2; -4 3]
    @planar t2[-1 -2 -3; -4] := ττ[-3 3; 1 2] * t[-1 -2 1 2; -4 3]
    @planar t3[-1 -2 -3; -4] := τ[2 1; 3 -3] * t[-1 -2 1 2; -4 3]
    @planar t4[-1 -2 -3; -4] := τ'[3 2; -3 1] * t[-1 -2 1 2; -4 3]
    @test t1 ≈ t2
    @test t1 ≈ t3
    @test t1 ≈ t4

    ττ = copy(BraidingTensor(V1', V1))
    @planar t1[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * τ[1 2; -4 3]
    @planar t2[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * ττ[1 2; -4 3]
    @planar t3[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * τ[3 -4; 2 1]
    @planar t4[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * τ'[2 3; 1 -4]
    @test t1 ≈ t2
    @test t1 ≈ t3
    @test t1 ≈ t4

    ττ = copy(BraidingTensor(V1, V1'))
    @planar t1[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * τ'[1 2; -4 3]
    @planar t2[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * ττ'[1 2; -4 3]
    @planar t3[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * τ'[3 -4; 2 1]
    @planar t4[-1 -2 -3; -4] := t[-1 -2 -3 3; 1 2] * τ[2 3; 1 -4]
    @test t1 ≈ t2
    @test t1 ≈ t3
    @test t1 ≈ t4
end

@testset "@planar" verbose = true begin
    T = ComplexF64
    @testset "MPS networks" begin
        P = ℂ^2
        Vmps = ℂ^12
        Vmpo = ℂ^4

        # ∂AC
        # -------
        x = TensorMap(randn, T, Vmps ⊗ P ← Vmps)
        O = TensorMap(randn, T, Vmpo ⊗ P ← P ⊗ Vmpo)
        GL = TensorMap(randn, T, Vmps ⊗ Vmpo' ← Vmps)
        GR = TensorMap(randn, T, Vmps ⊗ Vmpo ← Vmps)

        x′ = force_planar(x)
        O′ = force_planar(O)
        GL′ = force_planar(GL)
        GR′ = force_planar(GR)

        @tensor y[-1 -2; -3] := GL[-1 2; 1] * x[1 3; 4] * O[2 -2; 3 5] * GR[4 5; -3]
        @planar y′[-1 -2; -3] := GL′[-1 2; 1] * x′[1 3; 4] * O′[2 -2; 3 5] * GR′[4 5; -3]
        @test force_planar(y) ≈ y′

        # ∂AC2
        # -------
        x2 = TensorMap(randn, T, Vmps ⊗ P ← Vmps ⊗ P')
        x2′ = force_planar(x2)
        @tensor contractcheck = true y2[-1 -2; -3 -4] := GL[-1 7; 6] * x2[6 5; 1 3] *
                                                         O[7 -2; 5 4] * O[4 -4; 3 2] *
                                                         GR[1 2; -3]
        @planar y2′[-1 -2; -3 -4] := GL′[-1 7; 6] * x2′[6 5; 1 3] * O′[7 -2; 5 4] *
                                     O′[4 -4; 3 2] * GR′[1 2; -3]
        @test force_planar(y2) ≈ y2′

        # transfer matrix
        # ----------------
        v = TensorMap(randn, T, Vmps ← Vmps)
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
    end

    @testset "MERA networks" begin
        Vmera = ℂ^2

        u = TensorMap(randn, T, Vmera ⊗ Vmera ← Vmera ⊗ Vmera)
        w = TensorMap(randn, T, Vmera ⊗ Vmera ← Vmera)
        ρ = TensorMap(randn, T, Vmera ⊗ Vmera ⊗ Vmera ← Vmera ⊗ Vmera ⊗ Vmera)
        h = TensorMap(randn, T, Vmera ⊗ Vmera ⊗ Vmera ← Vmera ⊗ Vmera ⊗ Vmera)

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
end
