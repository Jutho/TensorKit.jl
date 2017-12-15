Vtr = (ℂ^2,
        ℂ^3,
        ℂ^4,
        ℂ^5,
        ℂ^6)
Vℤ₂ = (ℂ[ℤ₂](0=>1, 1=>1),
        ℂ[ℤ₂](0=>2, 1=>5),
        ℂ[ℤ₂](0=>3, 1=>2),
        ℂ[ℤ₂](0=>2, 1=>3),
        ℂ[ℤ₂](0=>1, 1=>2))
Vℤ₃ = (ℂ[ℤ₃](0=>1, 1=>1, 2=>2),
        ℂ[ℤ₃](0=>2, 1=>4, 2=>3),
        ℂ[ℤ₃](0=>3, 1=>2, 2=>1),
        ℂ[ℤ₃](0=>2, 1=>3, 2=>1),
        ℂ[ℤ₃](0=>1, 1=>2, 2=>3))
VU₁ = (ℂ[U₁](0=>1, 1=>1, -1=>2),
        ℂ[U₁](0=>2, 1=>4, -1=>3),
        ℂ[U₁](0=>3, 1=>2, -1=>1),
        ℂ[U₁](0=>2, 1=>3, -1=>1),
        ℂ[U₁](0=>1, 1=>2, -1=>3))
VSU₂ = (ℂ[SU₂](0=>1, 1//2=>1, 1=>2),
        ℂ[SU₂](0=>2, 1//2=>4, 1=>3),
        ℂ[SU₂](0=>3, 1//2=>2, 1=>1),
        ℂ[SU₂](0=>2, 1//2=>3, 1=>1),
        ℂ[SU₂](0=>1, 1//2=>2, 1=>3))

@testset "Tensors with symmetry: $G" for (G,V) in ((Trivial, Vtr), (ℤ₂, Vℤ₂), (ℤ₃, Vℤ₃), (U₁, VU₁), (SU₂, VSU₂))
    V1, V2, V3, V4, V5 = V
    @testset "Basic tensor properties" begin
        W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
        for T in (Int, Float32, Float64, ComplexF32, ComplexF64, BigFloat)
            t = Tensor(zeros, T, W)
            @test eltype(t) == T
            @test vecnorm(t) == 0
            @test codomain(t) == W
            @test space(t) == W
            @test domain(t) == one(W)
        end
    end
    @testset "Basic linear algebra" begin
        W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            t = TensorMap(rand, T, W)
            @test eltype(t) == T
            @test codomain(t) == W.second
            @test domain(t) == W.first
            @test isa(@inferred(vecnorm(t)), real(T))
            @test vecnorm(t)^2 ≈ vecdot(t,t)
            α = rand(T)
            @test vecnorm(α*t) ≈ abs(α)*vecnorm(t)
            @test vecnorm(t+t, 2) ≈ 2*vecnorm(t, 2)
            @test vecnorm(t+t, 1) ≈ 2*vecnorm(t, 1)
            @test vecnorm(t+t, Inf) ≈ 2*vecnorm(t, Inf)
            p = 3*rand(Float64)
            @test vecnorm(t+t, p) ≈ 2*vecnorm(t, p)

            t2 = TensorMap(rand, T, W)
            β = rand(T)
            @test vecdot(β*t2,α*t) ≈ conj(β)*α*conj(vecdot(t,t2))
        end
    end
    @testset "Permutations and inner product invariance" begin
        using Combinatorics
        W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
        t = Tensor(rand, W);
        t′ = Tensor(rand, W);
        for k = 0:5
            for p in permutations(1:5)
                p1 = ntuple(n->p[n], StaticLength(k))
                p2 = ntuple(n->p[k+n], StaticLength(5-k))
                t2 = @inferred permuteind(t, p1, p2)
                @test vecnorm(t2) ≈ vecnorm(t)
                t2′= permuteind(t′, p1, p2)
                @test vecdot(t2′,t2) ≈ vecdot(t′,t)
            end
        end
    end
    @testset "Factorization" begin
        W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            t = Tensor(rand, T, W)
            @testset "leftorth with $alg" for alg in (QR(), QRpos(), QL(), QLpos(), Polar(), SVD())
                Q, R = @inferred leftorth(t, (3,4,2),(1,5), alg)
                QdQ = Q'*Q
                @test QdQ ≈ one(QdQ)
                @test Q*R ≈ permuteind(t, (3,4,2),(1,5))
            end
            @testset "leftnull with $alg" for alg in (QR(), QRpos())
                N = @inferred leftnull(t, (3,4,2),(1,5), alg)
                NdN = N'*N
                @test NdN ≈ one(NdN)
                @test vecnorm(N'*permuteind(t, (3,4,2),(1,5))) < 100*eps(vecnorm(t))
            end
            @testset "rightorth with $alg" for alg in (RQ(), RQpos(), LQ(), LQpos(), Polar(), SVD())
                L, Q = @inferred rightorth(t, (3,4),(2,1,5), alg)
                QQd = Q*Q'
                @test QQd ≈ one(QQd)
                @test L*Q ≈ permuteind(t, (3,4),(2,1,5))
            end
            @testset "rightnull with $alg" for alg in (LQ(), LQpos())
                M = @inferred rightnull(t, (3,4),(2,1,5), alg)
                MMd = M*M'
                @test MMd ≈ one(MMd)
                @test vecnorm(permuteind(t, (3,4),(2,1,5))*M') < 100*eps(vecnorm(t))
            end
            U, S, V = @inferred svd(t, (3,4,2),(1,5))
            @test U*S*V ≈ permuteind(t, (3,4,2),(1,5))
        end
    end
end
