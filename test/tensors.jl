Vtr = (ℂ^3,
        (ℂ^4)',
        ℂ^5,
        ℂ^6,
        (ℂ^7)')
Vℤ₂ = (ℂ[ℤ₂](0=>1, 1=>1),
        ℂ[ℤ₂](0=>1, 1=>2)',
        ℂ[ℤ₂](0=>3, 1=>2)',
        ℂ[ℤ₂](0=>2, 1=>3),
        ℂ[ℤ₂](0=>2, 1=>5))
Vℤ₃ = (ℂ[ℤ₃](0=>1, 1=>2, 2=>2),
        ℂ[ℤ₃](0=>3, 1=>1, 2=>1),
        ℂ[ℤ₃](0=>2, 1=>2, 2=>1)',
        ℂ[ℤ₃](0=>1, 1=>2, 2=>3),
        ℂ[ℤ₃](0=>1, 1=>3, 2=>3)')
VU₁ = (ℂ[U₁](0=>1, 1=>2, -1=>2),
        ℂ[U₁](0=>3, 1=>1, -1=>1),
        ℂ[U₁](0=>2, 1=>2, -1=>1)',
        ℂ[U₁](0=>1, 1=>2, -1=>3),
        ℂ[U₁](0=>1, 1=>3, -1=>3)')
VCU₁ = (ℂ[CU₁]((0,0)=>1, (0,1)=>2, 1=>1),
        ℂ[CU₁]((0,0)=>3, (0,1)=>0, 1=>1),
        ℂ[CU₁]((0,0)=>1, (0,1)=>0, 1=>2)',
        ℂ[CU₁]((0,0)=>2, (0,1)=>2, 1=>1),
        ℂ[CU₁]((0,0)=>2, (0,1)=>1, 1=>2)')
VSU₂ = (ℂ[SU₂](0=>3, 1//2=>1),
        ℂ[SU₂](0=>2, 1=>1),
        ℂ[SU₂](1//2=>1, 1=>1)',
        ℂ[SU₂](0=>2, 1//2=>2),
        ℂ[SU₂](0=>1, 1//2=>1, 3//2=>1)')

for (G,V) in ((Trivial, Vtr), (ℤ₂, Vℤ₂), (ℤ₃, Vℤ₃), (U₁, VU₁), (CU₁, VCU₁), (SU₂, VSU₂))
    println("------------------------------------")
    println("Tensors with symmetry: $G")
    println("------------------------------------")
    ti = time()
    V1, V2, V3, V4, V5 = V
    @testset TimedTestSet "Basic tensor properties" begin
        W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
        for T in (Int, Float32, Float64, ComplexF32, ComplexF64, BigFloat)
            t = Tensor(zeros, T, W)
            @test @inferred(hash(t)) == hash(deepcopy(t))
            @test eltype(t) == T
            @test norm(t) == 0
            @test codomain(t) == W
            @test space(t) == (W ← one(W))
            @test domain(t) == one(W)
        end
    end
    @testset TimedTestSet "Tensor Dict conversion" begin
    W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
        for T in (Int, Float32, ComplexF64)
            t = TensorMap(rand, T, W)
            d = convert(Dict, t)
            @test t == convert(TensorMap, d)
        end
    end
    @testset TimedTestSet "Basic linear algebra" begin
        W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
        for T in (Float32, ComplexF64)
            t = TensorMap(rand, T, W)
            @test eltype(t) == T
            @test space(t) == W
            @test space(t') == W'
            @test dim(t) == dim(space(t))
            @test codomain(t) == codomain(W)
            @test domain(t) == domain(W)
            @test isa(@inferred(norm(t)), real(T))
            @test norm(t)^2 ≈ dot(t,t)
            α = rand(T)
            @test norm(α*t) ≈ abs(α)*norm(t)
            @test norm(t+t, 2) ≈ 2*norm(t, 2)
            @test norm(t+t, 1) ≈ 2*norm(t, 1)
            @test norm(t+t, Inf) ≈ 2*norm(t, Inf)
            p = 3*rand(Float64)
            @test norm(t+t, p) ≈ 2*norm(t, p)
            @test norm(t) ≈ norm(t')

            t2 = TensorMap(rand, T, W)
            β = rand(T)
            @test dot(β*t2,α*t) ≈ conj(β)*α*conj(dot(t,t2))
            @test dot(t2,t) ≈ conj(dot(t, t2))
            @test dot(t2,t) ≈ conj(dot(t2', t'))
            @test dot(t2,t) ≈ dot(t', t2')

            i1 = @inferred(isomorphism(Matrix{T}, V1 ⊗ V2, V2 ⊗ V1))
            i2 = @inferred(isomorphism(Matrix{T}, V2 ⊗ V1, V1 ⊗ V2))
            @test i1 * i2 == @inferred(id(Matrix{T}, V1 ⊗ V2))
            @test i2 * i1 == @inferred(id(Matrix{T}, V2 ⊗ V1))


            w = @inferred(isometry(Matrix{T}, V1 ⊗ (oneunit(V1) ⊕ oneunit(V1)), V1))
            @test dim(w) == 2*dim(V1←V1)
            @test w'*w == id(Matrix{T}, V1)
            @test w*w' == (w*w')^2
        end
    end
    @testset TimedTestSet "Basic linear algebra: test via conversion" begin
        W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
        for T in (Float32, ComplexF64)
            t = TensorMap(rand, T, W)
            t2 = TensorMap(rand, T, W)
            @test norm(t, 2) ≈ norm(convert(Array,t), 2)
            @test dot(t2,t) ≈ dot(convert(Array,t2), convert(Array, t))
            α = rand(T)
            @test convert(Array, α*t) ≈ α*convert(Array,t)
            @test convert(Array, t+t) ≈ 2*convert(Array,t)
        end
    end
    @testset TimedTestSet "Real and imaginary parts" begin
        W = V1 ⊗ V2
        for T in (Float64, ComplexF64, ComplexF32)
            t = TensorMap(randn, T, W, W)
            @test real(convert(Array, t)) == convert(Array, @inferred real(t))
            @test imag(convert(Array, t)) == convert(Array, @inferred imag(t))
        end
    end
    @testset TimedTestSet "Permutations: test via inner product invariance" begin
        W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
        t = Tensor(rand, ComplexF64, W);
        t′ = Tensor(rand, ComplexF64, W);
        for k = 0:5
            for p in permutations(1:5)
                p1 = ntuple(n->p[n], StaticLength(k))
                p2 = ntuple(n->p[k+n], StaticLength(5-k))
                t2 = @inferred permute(t, p1, p2)
                @test norm(t2) ≈ norm(t)
                t2′= permute(t′, p1, p2)
                @test dot(t2′,t2) ≈ dot(t′,t) ≈ dot(transpose(t2′), transpose(t2))
            end
        end
    end
    @testset TimedTestSet "Permutations: test via conversion" begin
        W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
        t = Tensor(rand, ComplexF64, W);
        for k = 0:5
            for p in permutations(1:5)
                p1 = ntuple(n->p[n], StaticLength(k))
                p2 = ntuple(n->p[k+n], StaticLength(5-k))
                t2 = permute(t, p1, p2)
                a2 = convert(Array, t2)
                @test a2 ≈ permutedims(convert(Array, t), (p1...,p2...))
                @test convert(Array, transpose(t2)) ≈ permutedims(a2, (5,4,3,2,1))
            end
        end
    end
    @testset TimedTestSet "Full trace: test self-consistency" begin
        t = Tensor(rand, ComplexF64, V1 ⊗ V2' ⊗ V2 ⊗ V1');
        t2 = permute(t, (1,2), (4,3))
        s = @inferred tr(t2)
        @test conj(s) ≈ tr(t2')
        @tensor s2 = t[a,b,b,a]
        @tensor t3[a,b] := t[a,c,c,b]
        @tensor s3 = t3[a,a]
        @test s ≈ s2
        @test s ≈ s3
    end
    @testset TimedTestSet "Partial trace: test self-consistency" begin
        t = Tensor(rand, ComplexF64, V1 ⊗ V2' ⊗ V3 ⊗ V2 ⊗ V1' ⊗ V1);
        @tensor t2[a,b] := t[c,d,b,d,c,a]
        @tensor t4[a,b,c,d] := t[d,e,b,e,c,a]
        @tensor t5[a,b] := t4[a,b,c,c]
        @test t2 ≈ t5
    end
    @testset TimedTestSet "Trace: test via conversion" begin
        t = Tensor(rand, ComplexF64, V1 ⊗ V2' ⊗ V3 ⊗ V2 ⊗ V1' ⊗ V1);
        @tensor t2[a,b] := t[c,d,b,d,c,a]
        @tensor t3[a,b] := convert(Array, t)[c,d,b,d,c,a]
        @test t3 ≈ convert(Array, t2)
    end
    @testset TimedTestSet "Trace and contraction" begin
        t1 = Tensor(rand, ComplexF64, V1 ⊗ V2 ⊗ V3);
        t2 = Tensor(rand, ComplexF64, V2' ⊗ V4 ⊗ V1');
        t3 = t1 ⊗ t2
        @tensor ta[a,b] := t1[x,y,a]*t2[y,b,x]
        @tensor tb[a,b] := t3[x,y,a,y,b,x]
        @test ta ≈ tb
    end
    @testset TimedTestSet "Tensor contraction: test via conversion" begin
        A1 = TensorMap(randn, ComplexF64, V1'*V2', V3')
        A2 = TensorMap(randn, ComplexF64, V3*V4, V5)
        rhoL = TensorMap(randn, ComplexF64, V1, V1)
        rhoR = TensorMap(randn, ComplexF64, V5, V5)' # test adjoint tensor
        H = TensorMap(randn, ComplexF64, V2*V4, V2*V4)
        @tensor HrA12[a, s1, s2, c] := rhoL[a, a'] * conj(A1[a', t1, b]) *
            A2[b, t2, c'] * rhoR[c', c] * H[s1, s2, t1, t2]

        @tensor HrA12array[a, s1, s2, c] := convert(Array, rhoL)[a, a'] *
            conj(convert(Array, A1)[a', t1, b]) *
            convert(Array, A2)[b, t2, c'] *
            convert(Array, rhoR)[c', c] *
            convert(Array, H)[s1, s2, t1, t2]

        @test HrA12array ≈ convert(Array, HrA12)
    end
    @testset TimedTestSet "Multiplication and inverse: test compatibility" begin
        W1 = V1 ⊗ V2 ⊗ V3
        W2 = V4 ⊗ V5
        for T in (Float64, ComplexF64)
            t1 = TensorMap(rand, T, W1, W1)
            t2 = TensorMap(rand, T, W2, W2)
            t = TensorMap(rand, T, W1, W2)
            @test t1*(t1\t) ≈ t
            @test (t/t2)*t2 ≈ t
            @test t1\one(t1) ≈ inv(t1)
            @test one(t1)/t1 ≈ pinv(t1)
            @test_throws SpaceMismatch inv(t)
            @test_throws SpaceMismatch t2\t
            @test_throws SpaceMismatch t/t1
            tp = pinv(t)*t
            @test tp ≈ tp*tp
        end
    end
    @testset TimedTestSet "Multiplication and inverse: test via conversion" begin
        W1 = V1 ⊗ V2 ⊗ V3
        W2 = V4 ⊗ V5
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            t1 = TensorMap(rand, T, W1, W1)
            t2 = TensorMap(rand, T, W2, W2)
            t = TensorMap(rand, T, W1, W2)
            d1 = dim(W1)
            d2 = dim(W2)
            At1 = reshape(convert(Array, t1), d1, d1)
            At2 = reshape(convert(Array, t2), d2, d2)
            At = reshape(convert(Array, t), d1, d2)
            @test reshape(convert(Array, t1*t), d1, d2) ≈ At1*At
            @test reshape(convert(Array, t1'*t), d1, d2) ≈ At1'*At
            @test reshape(convert(Array, t2*t'), d2, d1) ≈ At2*At'
            @test reshape(convert(Array, t2'*t'), d2, d1) ≈ At2'*At'

            @test reshape(convert(Array, inv(t1)), d1, d1) ≈ inv(At1)
            @test reshape(convert(Array, pinv(t)), d2, d1) ≈ pinv(At)

            if T == Float32 || T == ComplexF32
                continue
            end

            @test reshape(convert(Array, t1\t), d1, d2) ≈ At1\At
            @test reshape(convert(Array, t1'\t), d1, d2) ≈ At1'\At
            @test reshape(convert(Array, t2\t'), d2, d1) ≈ At2\At'
            @test reshape(convert(Array, t2'\t'), d2, d1) ≈ At2'\At'

            @test reshape(convert(Array, t2/t), d2, d1) ≈ At2/At
            @test reshape(convert(Array, t2'/t), d2, d1) ≈ At2'/At
            @test reshape(convert(Array, t1/t'), d1, d2) ≈ At1/At'
            @test reshape(convert(Array, t1'/t'), d1, d2) ≈ At1'/At'
        end
    end
    @testset TimedTestSet "Factorization" begin
        W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
        for T in (Float32, ComplexF64)
            # Test both a normal tensor and an adjoint one.
            ts = (Tensor(rand, T, W), Tensor(rand, T, W)')
            for t in ts
                @testset "leftorth with $alg" for alg in (TensorKit.QR(), TensorKit.QRpos(), TensorKit.QL(), TensorKit.QLpos(), TensorKit.Polar(), TensorKit.SVD(), TensorKit.SDD())
                    Q, R = @inferred leftorth(t, (3,4,2),(1,5); alg = alg)
                    QdQ = Q'*Q
                    @test QdQ ≈ one(QdQ)
                    @test Q*R ≈ permute(t, (3,4,2),(1,5))
                end
                @testset "leftnull with $alg" for alg in (TensorKit.QR(), TensorKit.SVD(), TensorKit.SDD())
                    N = @inferred leftnull(t, (3,4,2),(1,5); alg = alg)
                    NdN = N'*N
                    @test NdN ≈ one(NdN)
                    @test norm(N'*permute(t, (3,4,2),(1,5))) < 100*eps(norm(t))
                end
                @testset "rightorth with $alg" for alg in (TensorKit.RQ(), TensorKit.RQpos(), TensorKit.LQ(), TensorKit.LQpos(), TensorKit.Polar(), TensorKit.SVD(), TensorKit.SDD())
                    L, Q = @inferred rightorth(t, (3,4),(2,1,5); alg = alg)
                    QQd = Q*Q'
                    @test QQd ≈ one(QQd)
                    @test L*Q ≈ permute(t, (3,4),(2,1,5))
                end
                @testset "rightnull with $alg" for alg in (TensorKit.LQ(), TensorKit.SVD(), TensorKit.SDD())
                    M = @inferred rightnull(t, (3,4),(2,1,5); alg = alg)
                    MMd = M*M'
                    @test MMd ≈ one(MMd)
                    @test norm(permute(t, (3,4),(2,1,5))*M') < 100*eps(norm(t))
                end
                @testset "tsvd with $alg" for alg in (TensorKit.SVD(), TensorKit.SDD())
                    U, S, V = @inferred tsvd(t, (3,4,2),(1,5); alg = alg)
                    UdU = U'*U
                    @test UdU ≈ one(UdU)
                    VVd = V*V'
                    @test VVd ≈ one(VVd)
                    @test U*S*V ≈ permute(t, (3,4,2),(1,5))
                end
            end
            @testset "empty tensor" begin
                t = TensorMap(randn, T, V1 ⊗ V2, typeof(V1)())
                @testset "leftorth with $alg" for alg in (TensorKit.QR(), TensorKit.QRpos(), TensorKit.QL(), TensorKit.QLpos(), TensorKit.Polar(), TensorKit.SVD(), TensorKit.SDD())
                    Q, R = @inferred leftorth(t; alg = alg)
                    @test Q == t
                    @test dim(Q) == dim(R) == 0
                end
                @testset "leftnull with $alg" for alg in (TensorKit.QR(), TensorKit.SVD(), TensorKit.SDD())
                    N = @inferred leftnull(t; alg = alg)
                    @test N'*N ≈ id(domain(N))
                    @test N*N' ≈ id(codomain(N))
                end
                @testset "rightorth with $alg" for alg in (TensorKit.RQ(), TensorKit.RQpos(), TensorKit.LQ(), TensorKit.LQpos(), TensorKit.Polar(), TensorKit.SVD(), TensorKit.SDD())
                    L, Q = @inferred rightorth(copy(t'); alg = alg)
                    @test Q == t'
                    @test dim(Q) == dim(L) == 0
                end
                @testset "rightnull with $alg" for alg in (TensorKit.LQ(), TensorKit.SVD(), TensorKit.SDD())
                    M = @inferred rightnull(copy(t'); alg = alg)
                    @test M*M' ≈ id(codomain(M))
                    @test M'*M ≈ id(domain(M))
                end
                @testset "tsvd with $alg" for alg in (TensorKit.SVD(), TensorKit.SDD())
                    U, S, V = @inferred tsvd(t; alg = alg)
                    @test U == t
                    @test dim(U) == dim(S) == dim(V)
                end
            end

            t = Tensor(rand, T, V1 ⊗ V1' ⊗ V2 ⊗ V2')
            @testset "eig and isposdef" begin
                D, V = eigen(t, (1,3), (2,4))
                VdV = V'*V
                VdV = (VdV + VdV')/2
                @test isposdef(VdV)
                t2 = permute(t, (1,3), (2,4))
                @test t2*V ≈ V*D
                @test !isposdef(t2) # unlikely for non-hermitian map
                t2 = (t2 + t2');
                D, V = eigen(t2)
                VdV = V'*V
                @test VdV ≈ one(VdV)
                λ = minimum(minimum(real(diag(b))) for (c,b) in blocks(D))
                @test isposdef(t2) == isposdef(λ)
                @test isposdef(t2 - λ*one(t2) + 0.1*one(t2))
                @test !isposdef(t2 - λ*one(t2) - 0.1*one(t2))
            end
        end
    end
    @testset TimedTestSet "Tensor truncation" begin
        for T in (Float32, ComplexF64)
            for p in (1, 2, 3, Inf)
                # Test both a normal tensor and an adjoint one.
                ts = (TensorMap(randn, T, V1 ⊗ V2 ⊗ V3, V4 ⊗ V5),
                      TensorMap(randn, T, V4 ⊗ V5, V1 ⊗ V2 ⊗ V3)')
                for t in ts
                    U₀, S₀, V₀, = tsvd(t)
                    t = rmul!(t, 1/norm(S₀, p))
                    U, S, V, ϵ = @inferred tsvd(t; trunc = truncerr(5e-1), p = p)
                    # @show p, ϵ
                    # @show domain(S)
                    # @test min(space(S,1), space(S₀,1)) != space(S₀,1)
                    U′, S′, V′, ϵ′ = tsvd(t; trunc = truncerr(nextfloat(ϵ)), p = p)
                    @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
                    U′, S′, V′, ϵ′ = tsvd(t; trunc = truncdim(dim(domain(S))), p = p)
                    @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
                    U′, S′, V′, ϵ′ = tsvd(t; trunc = truncspace(space(S,1)), p = p)
                    @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
                    # results with truncationcutoff cannot be compared because they don't take degeneracy into account, and thus truncate differently
                    U, S, V, ϵ = tsvd(t; trunc = truncbelow(1/dim(domain(S₀))), p = p)
                    # @show p, ϵ
                    # @show domain(S)
                    # @test min(space(S,1), space(S₀,1)) != space(S₀,1)
                    U′, S′, V′, ϵ′ = tsvd(t; trunc = truncspace(space(S,1)), p = p)
                    @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
                end
            end
        end
    end
    @testset TimedTestSet "Tensor functions" begin
        W = V1 ⊗ V2
        for T in (Float64, ComplexF64)
            t = TensorMap(randn, T, W, W)
            s = dim(W)
            expt = @inferred exp(t)
            @test reshape(convert(Array, expt), (s,s)) ≈
                    exp(reshape(convert(Array, t), (s,s)))

            @test (@inferred sqrt(t))^2 ≈ t
            @test reshape(convert(Array, sqrt(t^2)), (s,s)) ≈
                    sqrt(reshape(convert(Array, t^2), (s,s)))

            @test exp(@inferred log(expt)) ≈ expt
            @test reshape(convert(Array, log(expt)), (s,s)) ≈
                    log(reshape(convert(Array, expt), (s,s)))

            @test (@inferred cos(t))^2 + (@inferred sin(t))^2 ≈ id(W)
            @test (@inferred tan(t)) ≈ sin(t)/cos(t)
            @test (@inferred cot(t)) ≈ cos(t)/sin(t)
            @test (@inferred cosh(t))^2 - (@inferred sinh(t))^2 ≈ id(W)
            @test (@inferred tanh(t)) ≈ sinh(t)/cosh(t)
            @test (@inferred coth(t)) ≈ cosh(t)/sinh(t)

            t1 = sin(t)
            @test sin(@inferred asin(t1)) ≈ t1
            t2 = cos(t)
            @test cos(@inferred acos(t2)) ≈ t2
            t3 = sinh(t)
            @test sinh(@inferred asinh(t3)) ≈ t3
            t4 = cosh(t)
            @test cosh(@inferred acosh(t4)) ≈ t4
            t5 = tan(t)
            @test tan(@inferred atan(t5)) ≈ t5
            t6 = cot(t)
            @test cot(@inferred acot(t6)) ≈ t6
            t7 = tanh(t)
            @test tanh(@inferred atanh(t7)) ≈ t7
            t8 = coth(t)
            @test coth(@inferred acoth(t8)) ≈ t8
        end
    end
    @testset TimedTestSet "Sylvester equation" begin
        for T in (Float32, ComplexF64)
            tA = TensorMap(rand, T, V1 ⊗ V3, V1 ⊗ V3)
            tB = TensorMap(rand, T, V2 ⊗ V4, V2 ⊗ V4)
            tC = TensorMap(rand, T, V1 ⊗ V3, V2 ⊗ V4)
            t = @inferred sylvester(tA, tB, tC)
            @test codomain(t) == V1 ⊗ V3
            @test domain(t) == V2 ⊗ V4
            @test norm(tA*t + t*tB + tC) < sqrt(eps(real(T)))
            matrix(x) = reshape(convert(Array, x), dim(codomain(x)), dim(domain(x)))
            @test matrix(t) ≈ sylvester(matrix(tA), matrix(tB), matrix(tC))
        end
    end
    @testset TimedTestSet "Tensor product: test via norm preservation" begin
        for T in (Float32, ComplexF64)
            t1 = TensorMap(rand, T, V2 ⊗ V3 ⊗ V1, V1 ⊗ V2)
            t2 = TensorMap(rand, T, V2 ⊗ V1 ⊗ V3, V1 ⊗ V1)
            t = @inferred (t1 ⊗ t2)
            @test norm(t) ≈ norm(t1) * norm(t2)
        end
    end
    @testset TimedTestSet "Tensor product: test via conversion" begin
        for T in (Float32, ComplexF64)
            t1 = TensorMap(rand, T, V2 ⊗ V3 ⊗ V1, V1)
            t2 = TensorMap(rand, T, V2 ⊗ V1 ⊗ V3, V2)
            t = @inferred (t1 ⊗ t2)
            d1 = dim(codomain(t1))
            d2 = dim(codomain(t2))
            d3 = dim(domain(t1))
            d4 = dim(domain(t2))
            At = convert(Array, t)
            @test reshape(At, (d1, d2, d3, d4)) ≈
                    reshape(convert(Array, t1), (d1, 1, d3, 1)) .*
                    reshape(convert(Array, t2), (1, d2, 1, d4))
        end
    end
    @testset TimedTestSet "Tensor product: test via tensor contraction" begin
        for T in (Float32, ComplexF64)
            t1 = Tensor(rand, T, V2 ⊗ V3 ⊗ V1)
            t2 = Tensor(rand, T, V2 ⊗ V1 ⊗ V3)
            t = @inferred (t1 ⊗ t2)
            @tensor t′[1, 2, 3, 4, 5, 6] := t1[1,2,3]*t2[4,5,6]
            @test t ≈ t′
        end
    end
    tf = time()
    printstyled("Finished tensor tests with symmetry $G in ",
                string(round(tf-ti; sigdigits=3)),
                " seconds."; bold = true, color = Base.info_color())
    println()
end
