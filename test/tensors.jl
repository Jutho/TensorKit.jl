Vtr = (ℂ^3,
        (ℂ^4)',
        ℂ^5,
        ℂ^6,
        (ℂ^7)')
Vℤ₂ = (ℂ[Z2Irrep](0=>1, 1=>1),
        ℂ[Z2Irrep](0=>1, 1=>2)',
        ℂ[Z2Irrep](0=>3, 1=>2)',
        ℂ[Z2Irrep](0=>2, 1=>3),
        ℂ[Z2Irrep](0=>2, 1=>5))
Vℤ₃ = (ℂ[Z3Irrep](0=>1, 1=>2, 2=>2),
        ℂ[Z3Irrep](0=>3, 1=>1, 2=>1),
        ℂ[Z3Irrep](0=>2, 1=>2, 2=>1)',
        ℂ[Z3Irrep](0=>1, 1=>2, 2=>3),
        ℂ[Z3Irrep](0=>1, 1=>3, 2=>3)')
VU₁ = (ℂ[U1Irrep](0=>1, 1=>2, -1=>2),
        ℂ[U1Irrep](0=>3, 1=>1, -1=>1),
        ℂ[U1Irrep](0=>2, 1=>2, -1=>1)',
        ℂ[U1Irrep](0=>1, 1=>2, -1=>3),
        ℂ[U1Irrep](0=>1, 1=>3, -1=>3)')
VCU₁ = (ℂ[CU1Irrep]((0,0)=>1, (0,1)=>2, 1=>1),
        ℂ[CU1Irrep]((0,0)=>3, (0,1)=>0, 1=>1),
        ℂ[CU1Irrep]((0,0)=>1, (0,1)=>0, 1=>2)',
        ℂ[CU1Irrep]((0,0)=>2, (0,1)=>2, 1=>1),
        ℂ[CU1Irrep]((0,0)=>2, (0,1)=>1, 1=>2)')
VSU₂ = (ℂ[SU2Irrep](0=>3, 1//2=>1),
        ℂ[SU2Irrep](0=>2, 1=>1),
        ℂ[SU2Irrep](1//2=>1, 1=>1)',
        ℂ[SU2Irrep](0=>2, 1//2=>2),
        ℂ[SU2Irrep](0=>1, 1//2=>1, 3//2=>1)')
VSU₃ = (ℂ[SU3Irrep]((0,0,0)=>3, (1,0,0)=>1),
               ℂ[SU3Irrep]((0,0,0)=>3, (2,0,0)=>1)',
               ℂ[SU3Irrep]((1,1,0)=>1, (2,1,0)=>1),
               ℂ[SU3Irrep]((1,0,0)=>1, (2,0,0)=>1),
               ℂ[SU3Irrep]((0,0,0)=>1, (1,0,0)=>1, (1,1,0)=>1)')

for V in (Vtr, Vℤ₂, Vℤ₃, VU₁, VCU₁, VSU₂)#, VSU₃)
    V1, V2, V3, V4, V5 = V
    @assert V3 * V4 * V2 ≿ V1' * V5' # necessary for leftorth tests
    @assert V3 * V4 ≾ V1' * V2' * V5' # necessary for rightorth tests
end

for V in (Vtr, Vℤ₂, Vℤ₃, VU₁, VCU₁, VSU₂, VSU₃)
    I = sectortype(first(V))
    Istr = TensorKit.type_repr(I)
    println("---------------------------------------")
    println("Tensors with symmetry: $Istr")
    println("---------------------------------------")
    global ti = time()
    V1, V2, V3, V4, V5 = V
    @timedtestset "Basic tensor properties" begin
        W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
        for T in (Int, Float32, Float64, ComplexF32, ComplexF64, BigFloat)
            t = Tensor(zeros, T, W)
            @test @constinferred(hash(t)) == hash(deepcopy(t))
            @test eltype(t) == T
            @test norm(t) == 0
            @test codomain(t) == W
            @test space(t) == (W ← one(W))
            @test domain(t) == one(W)
            @test typeof(t) == @constinferred tensormaptype(spacetype(t), 5, 0, T)
        end
    end
    @timedtestset "Tensor Dict conversion" begin
        W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
        for T in (Int, Float32, ComplexF64)
            t = TensorMap(rand, T, W)
            d = convert(Dict, t)
            @test t == convert(TensorMap, d)
        end
    end
    if hasfusiontensor(I)
        @timedtestset "Tensor Array conversion" begin
            W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
            for T in (Int, Float32, ComplexF64)
                if T == Int
                    t = TensorMap(sz->rand(-20:20, sz), W)
                else
                    t = TensorMap(randn, T, W)
                end
                a = @constinferred convert(Array, t)
                @test t ≈ @constinferred TensorMap(a, W)
            end
        end
    end
    @timedtestset "Basic linear algebra" begin
        W = V1 ⊗ V2 ⊗ V3 ← V4 ⊗ V5
        for T in (Float32, ComplexF64)
            t = TensorMap(rand, T, W)
            @test eltype(t) == T
            @test space(t) == W
            @test space(t') == W'
            @test dim(t) == dim(space(t))
            @test codomain(t) == codomain(W)
            @test domain(t) == domain(W)
            @test isa(@constinferred(norm(t)), real(T))
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
            @test @constinferred(dot(β*t2,α*t)) ≈ conj(β)*α*conj(dot(t,t2))
            @test dot(t2,t) ≈ conj(dot(t, t2))
            @test dot(t2,t) ≈ conj(dot(t2', t'))
            @test dot(t2,t) ≈ dot(t', t2')

            i1 = @constinferred(isomorphism(Matrix{T}, V1 ⊗ V2, V2 ⊗ V1))
            i2 = @constinferred(isomorphism(Matrix{T}, V2 ⊗ V1, V1 ⊗ V2))
            @test i1 * i2 == @constinferred(id(Matrix{T}, V1 ⊗ V2))
            @test i2 * i1 == @constinferred(id(Matrix{T}, V2 ⊗ V1))


            w = @constinferred(isometry(Matrix{T}, V1 ⊗ (oneunit(V1) ⊕ oneunit(V1)), V1))
            @test dim(w) == 2*dim(V1←V1)
            @test w'*w == id(Matrix{T}, V1)
            @test w*w' == (w*w')^2
        end
    end
    if hasfusiontensor(I)
        @timedtestset "Basic linear algebra: test via conversion" begin
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
        @timedtestset "Real and imaginary parts" begin
            W = V1 ⊗ V2
            for T in (Float64, ComplexF64, ComplexF32)
                t = TensorMap(randn, T, W, W)
                @test real(convert(Array, t)) == convert(Array, @constinferred real(t))
                @test imag(convert(Array, t)) == convert(Array, @constinferred imag(t))
            end
        end
    end
    @timedtestset "Tensor conversion" begin
        W = V1 ⊗ V2
        t = TensorMap(randn, Float64, W, W)
        @test typeof(convert(TensorMap, t')) == typeof(t)
        tc = complex(t)
        @test convert(typeof(tc), t) == tc
        @test typeof(convert(typeof(tc), t)) == typeof(tc)
        @test typeof(convert(typeof(tc), t')) == typeof(tc)
    end
    @timedtestset "Permutations: test via inner product invariance" begin
        W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
        t = Tensor(rand, ComplexF64, W)
        t′ = Tensor(rand, ComplexF64, W)
        for k = 0:5
            for p in permutations(1:5)
                p1 = ntuple(n->p[n], k)
                p2 = ntuple(n->p[k+n], 5-k)
                t2 = @constinferred permute(t, p1, p2)
                @test norm(t2) ≈ norm(t)
                t2′= permute(t′, p1, p2)
                @test dot(t2′,t2) ≈ dot(t′,t) ≈ dot(transpose(t2′), transpose(t2))
            end
        end
    end
    if hasfusiontensor(I)
        @timedtestset "Permutations: test via conversion" begin
            W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
            t = Tensor(rand, ComplexF64, W)
            for k = 0:5
                for p in permutations(1:5)
                    p1 = ntuple(n->p[n], k)
                    p2 = ntuple(n->p[k+n], 5-k)
                    t2 = permute(t, p1, p2)
                    a2 = convert(Array, t2)
                    @test a2 ≈ permutedims(convert(Array, t), (p1...,p2...))
                    @test convert(Array, transpose(t2)) ≈ permutedims(a2, (5,4,3,2,1))
                end
            end
        end
    end
    @timedtestset "Full trace: test self-consistency" begin
        t = Tensor(rand, ComplexF64, V1 ⊗ V2' ⊗ V2 ⊗ V1')
        t2 = permute(t, (1,2), (4,3))
        s = @constinferred tr(t2)
        @test conj(s) ≈ tr(t2')
        @tensor s2 = t[a,b,b,a]
        @tensor t3[a,b] := t[a,c,c,b]
        @tensor s3 = t3[a,a]
        @test s ≈ s2
        @test s ≈ s3
    end
    @timedtestset "Partial trace: test self-consistency" begin
        t = Tensor(rand, ComplexF64, V1 ⊗ V2' ⊗ V3 ⊗ V2 ⊗ V1' ⊗ V3')
        @tensor t2[a,b] := t[c,d,b,d,c,a]
        @tensor t4[a,b,c,d] := t[d,e,b,e,c,a]
        @tensor t5[a,b] := t4[a,b,c,c]
        @test t2 ≈ t5
    end
    if hasfusiontensor(I)
        @timedtestset "Trace: test via conversion" begin
            t = Tensor(rand, ComplexF64, V1 ⊗ V2' ⊗ V3 ⊗ V2 ⊗ V1' ⊗ V3')
            @tensor t2[a,b] := t[c,d,b,d,c,a]
            @tensor t3[a,b] := convert(Array, t)[c,d,b,d,c,a]
            @test t3 ≈ convert(Array, t2)
        end
    end
    @timedtestset "Trace and contraction" begin
        t1 = Tensor(rand, ComplexF64, V1 ⊗ V2 ⊗ V3)
        t2 = Tensor(rand, ComplexF64, V2' ⊗ V4 ⊗ V1')
        t3 = t1 ⊗ t2
        @tensor ta[a,b] := t1[x,y,a]*t2[y,b,x]
        @tensor tb[a,b] := t3[x,y,a,y,b,x]
        @test ta ≈ tb
    end
    if hasfusiontensor(I)
        @timedtestset "Tensor contraction: test via conversion" begin
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
    end
    @timedtestset "Multiplication and inverse: test compatibility" begin
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
    if hasfusiontensor(I)
        @timedtestset "Multiplication and inverse: test via conversion" begin
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
    end
    @timedtestset "Factorization" begin
        W = V1 ⊗ V2 ⊗ V3 ⊗ V4 ⊗ V5
        for T in (Float32, ComplexF64)
            # Test both a normal tensor and an adjoint one.
            ts = (Tensor(rand, T, W), Tensor(rand, T, W)')
            for t in ts
                @testset "leftorth with $alg" for alg in (TensorKit.QR(), TensorKit.QRpos(), TensorKit.QL(), TensorKit.QLpos(), TensorKit.Polar(), TensorKit.SVD(), TensorKit.SDD())
                    Q, R = @constinferred leftorth(t, (3,4,2), (1,5); alg = alg)
                    QdQ = Q'*Q
                    @test QdQ ≈ one(QdQ)
                    @test Q*R ≈ permute(t, (3,4,2), (1,5))
                    if alg isa Polar
                        @test isposdef(R)
                        @test domain(R) == codomain(R) == space(t, 1)' ⊗ space(t, 5)'
                    end
                end
                @testset "leftnull with $alg" for alg in (TensorKit.QR(), TensorKit.SVD(), TensorKit.SDD())
                    N = @constinferred leftnull(t, (3,4,2),(1,5); alg = alg)
                    NdN = N'*N
                    @test NdN ≈ one(NdN)
                    @test norm(N'*permute(t, (3,4,2),(1,5))) < 100*eps(norm(t))
                end
                @testset "rightorth with $alg" for alg in (TensorKit.RQ(), TensorKit.RQpos(), TensorKit.LQ(), TensorKit.LQpos(), TensorKit.Polar(), TensorKit.SVD(), TensorKit.SDD())
                    L, Q = @constinferred rightorth(t, (3,4),(2,1,5); alg = alg)
                    QQd = Q*Q'
                    @test QQd ≈ one(QQd)
                    @test L*Q ≈ permute(t, (3,4),(2,1,5))
                    if alg isa Polar
                        @test isposdef(L)
                        @test domain(L) == codomain(L) == space(t, 3) ⊗ space(t, 4)
                    end
                end
                @testset "rightnull with $alg" for alg in (TensorKit.LQ(), TensorKit.SVD(), TensorKit.SDD())
                    M = @constinferred rightnull(t, (3,4),(2,1,5); alg = alg)
                    MMd = M*M'
                    @test MMd ≈ one(MMd)
                    @test norm(permute(t, (3,4),(2,1,5))*M') < 100*eps(norm(t))
                end
                @testset "tsvd with $alg" for alg in (TensorKit.SVD(), TensorKit.SDD())
                    U, S, V = @constinferred tsvd(t, (3,4,2),(1,5); alg = alg)
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
                    Q, R = @constinferred leftorth(t; alg = alg)
                    @test Q == t
                    @test dim(Q) == dim(R) == 0
                end
                @testset "leftnull with $alg" for alg in (TensorKit.QR(), TensorKit.SVD(), TensorKit.SDD())
                    N = @constinferred leftnull(t; alg = alg)
                    @test N'*N ≈ id(domain(N))
                    @test N*N' ≈ id(codomain(N))
                end
                @testset "rightorth with $alg" for alg in (TensorKit.RQ(), TensorKit.RQpos(), TensorKit.LQ(), TensorKit.LQpos(), TensorKit.Polar(), TensorKit.SVD(), TensorKit.SDD())
                    L, Q = @constinferred rightorth(copy(t'); alg = alg)
                    @test Q == t'
                    @test dim(Q) == dim(L) == 0
                end
                @testset "rightnull with $alg" for alg in (TensorKit.LQ(), TensorKit.SVD(), TensorKit.SDD())
                    M = @constinferred rightnull(copy(t'); alg = alg)
                    @test M*M' ≈ id(codomain(M))
                    @test M'*M ≈ id(domain(M))
                end
                @testset "tsvd with $alg" for alg in (TensorKit.SVD(), TensorKit.SDD())
                    U, S, V = @constinferred tsvd(t; alg = alg)
                    @test U == t
                    @test dim(U) == dim(S) == dim(V)
                end
            end

            t = Tensor(rand, T, V1 ⊗ V1' ⊗ V2 ⊗ V2')
            @testset "eig and isposdef" begin
                D, V = eigen(t, (1,3), (2,4))
                D̃, Ṽ = @constinferred eig(t, (1,3), (2,4))
                @test D ≈ D̃
                @test V ≈ Ṽ
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
                D̃, Ṽ = @constinferred eigh(t2)
                @test D ≈ D̃
                @test V ≈ Ṽ
                λ = minimum(minimum(real(LinearAlgebra.diag(b))) for (c,b) in blocks(D))
                @test isposdef(t2) == isposdef(λ)
                @test isposdef(t2 - λ*one(t2) + 0.1*one(t2))
                @test !isposdef(t2 - λ*one(t2) - 0.1*one(t2))
            end
        end
    end
    @timedtestset "Tensor truncation" begin
        for T in (Float32, ComplexF64)
            for p in (1, 2, 3, Inf)
                # Test both a normal tensor and an adjoint one.
                ts = (TensorMap(randn, T, V1 ⊗ V2 ⊗ V3, V4 ⊗ V5),
                      TensorMap(randn, T, V4 ⊗ V5, V1 ⊗ V2 ⊗ V3)')
                for t in ts
                    U₀, S₀, V₀, = tsvd(t)
                    t = rmul!(t, 1/norm(S₀, p))
                    U, S, V, ϵ = @constinferred tsvd(t; trunc = truncerr(5e-1), p = p)
                    # @show p, ϵ
                    # @show domain(S)
                    # @test min(space(S,1), space(S₀,1)) != space(S₀,1)
                    U′, S′, V′, ϵ′ = tsvd(t; trunc = truncerr(nextfloat(ϵ)), p = p)
                    @test (U, S, V, ϵ) == (U′, S′, V′, ϵ′)
                    U′, S′, V′, ϵ′ = tsvd(t; trunc = truncdim(ceil(Int, dim(domain(S)))), p = p)
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
    if hasfusiontensor(I)
        @timedtestset "Tensor functions" begin
            W = V1 ⊗ V2
            for T in (Float64, ComplexF64)
                t = TensorMap(randn, T, W, W)
                s = dim(W)
                expt = @constinferred exp(t)
                @test reshape(convert(Array, expt), (s,s)) ≈
                        exp(reshape(convert(Array, t), (s,s)))

                @test (@constinferred sqrt(t))^2 ≈ t
                @test reshape(convert(Array, sqrt(t^2)), (s,s)) ≈
                        sqrt(reshape(convert(Array, t^2), (s,s)))

                @test exp(@constinferred log(expt)) ≈ expt
                @test reshape(convert(Array, log(expt)), (s,s)) ≈
                        log(reshape(convert(Array, expt), (s,s)))

                @test (@constinferred cos(t))^2 + (@constinferred sin(t))^2 ≈ id(W)
                @test (@constinferred tan(t)) ≈ sin(t)/cos(t)
                @test (@constinferred cot(t)) ≈ cos(t)/sin(t)
                @test (@constinferred cosh(t))^2 - (@constinferred sinh(t))^2 ≈ id(W)
                @test (@constinferred tanh(t)) ≈ sinh(t)/cosh(t)
                @test (@constinferred coth(t)) ≈ cosh(t)/sinh(t)

                t1 = sin(t)
                @test sin(@constinferred asin(t1)) ≈ t1
                t2 = cos(t)
                @test cos(@constinferred acos(t2)) ≈ t2
                t3 = sinh(t)
                @test sinh(@constinferred asinh(t3)) ≈ t3
                t4 = cosh(t)
                @test cosh(@constinferred acosh(t4)) ≈ t4
                t5 = tan(t)
                @test tan(@constinferred atan(t5)) ≈ t5
                t6 = cot(t)
                @test cot(@constinferred acot(t6)) ≈ t6
                t7 = tanh(t)
                @test tanh(@constinferred atanh(t7)) ≈ t7
                t8 = coth(t)
                @test coth(@constinferred acoth(t8)) ≈ t8
            end
        end
    end
    @timedtestset "Sylvester equation" begin
        for T in (Float32, ComplexF64)
            tA = TensorMap(rand, T, V1 ⊗ V3, V1 ⊗ V3)
            tB = TensorMap(rand, T, V2 ⊗ V4, V2 ⊗ V4)
            tA = tA*tA'
            tB = tB*tB'
            tC = TensorMap(rand, T, V1 ⊗ V3, V2 ⊗ V4)
            t = @constinferred sylvester(tA, tB, tC)
            @test codomain(t) == V1 ⊗ V3
            @test domain(t) == V2 ⊗ V4
            # @test norm(tA*t + t*tB + tC) < (norm(tA)+norm(tB)+norm(tC))*eps(real(T))
            if hasfusiontensor(I)
                matrix(x) = reshape(convert(Array, x), dim(codomain(x)), dim(domain(x)))
                @test matrix(t) ≈ sylvester(matrix(tA), matrix(tB), matrix(tC))
            end
        end
    end
    @timedtestset "Tensor product: test via norm preservation" begin
        for T in (Float32, ComplexF64)
            t1 = TensorMap(rand, T, V2 ⊗ V3 ⊗ V1, V1 ⊗ V2)
            t2 = TensorMap(rand, T, V2 ⊗ V1 ⊗ V3, V1 ⊗ V1)
            t = @constinferred (t1 ⊗ t2)
            @test norm(t) ≈ norm(t1) * norm(t2)
        end
    end
    if hasfusiontensor(I)
        @timedtestset "Tensor product: test via conversion" begin
            for T in (Float32, ComplexF64)
                t1 = TensorMap(rand, T, V2 ⊗ V3 ⊗ V1, V1)
                t2 = TensorMap(rand, T, V2 ⊗ V1 ⊗ V3, V2)
                t = @constinferred (t1 ⊗ t2)
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
    end
    @timedtestset "Tensor product: test via tensor contraction" begin
        for T in (Float32, ComplexF64)
            t1 = Tensor(rand, T, V2 ⊗ V3 ⊗ V1)
            t2 = Tensor(rand, T, V2 ⊗ V1 ⊗ V3)
            t = @constinferred (t1 ⊗ t2)
            @tensor t′[1, 2, 3, 4, 5, 6] := t1[1,2,3]*t2[4,5,6]
            @test t ≈ t′
        end
    end
    global tf = time()
    printstyled("Finished tensor tests with symmetry $Istr in ",
                string(round(tf-ti; sigdigits=3)),
                " seconds."; bold = true, color = Base.info_color())
    println()
end

@timedtestset "Deligne tensor product: test via conversion" begin
    @testset for Vlist1 in (Vtr, VSU₂), Vlist2 in (Vtr, Vℤ₂)
        V1, V2, V3, V4, V5 = Vlist1
        W1, W2, W3, W4, W5 = Vlist2
        for T in (Float32, ComplexF64)
            t1 = TensorMap(rand, T, V1 ⊗ V2, V3' ⊗ V4)
            t2 = TensorMap(rand, T, W2, W1 ⊗ W1')
            t = @constinferred (t1 ⊠ t2)
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
end
