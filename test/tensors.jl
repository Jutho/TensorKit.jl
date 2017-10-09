@testset "ComplexTensor" begin
    W = ℂ^2 ⊗ ℂ^3 ⊗ ℂ^4 ⊗ ℂ^5 ⊗ ℂ^6
    for T in (Int, Float32, Float64, Complex64, Complex128, BigFloat)
        t = Tensor(zeros, T, W)
        @test eltype(t) == T
        @test vecnorm(t) == 0
    end
    for T in (Float32, Float64, Complex64, Complex128)
        t = Tensor(rand, T, W)
        @test eltype(t) == T
        @test isa(@inferred(vecnorm(t)), real(T))
        @test vecnorm(t)^2 ≈ vecdot(t,t)
        α = rand(T)
        @test vecnorm(α*t) ≈ abs(α)*vecnorm(t)
        @test vecnorm(t+t, 2) ≈ 2*vecnorm(t, 2)
        @test vecnorm(t+t, 1) ≈ 2*vecnorm(t, 1)
        @test vecnorm(t+t, Inf) ≈ 2*vecnorm(t, Inf)
        p = 3*rand(Float64)
        @test vecnorm(t+t, p) ≈ 2*vecnorm(t, p)
    end
    for T in (Float32, Float64, Complex64, Complex128)
        t = Tensor(rand, T, W)
        @inferred permuteind(t, (3,4,2,1,5))
        @inferred permuteind(t, (3,4),(2,1,5))
        Q , R = @inferred leftorth(t, (3,4,2),(1,5))
        @test vecnorm(Q*R - permuteind(t, (3,4,2),(1,5))) < 100*eps(real(T))
        N = @inferred leftnull(t, (3,4,2),(1,5))
        # @test vecnorm(N'*permuteind(t, (3,4,2),(1,5))) < 100*eps(real(T))
        L, Q = @inferred rightorth(t, (3,4),(2,1,5))
        @test vecnorm(L*Q - permuteind(t, (3,4),(2,1,5))) < 100*eps(real(T))
        M = @inferred rightnull(t, (3,4,2),(1,5))
        U, S, V = @inferred svd(t, (3,4,2),(1,5))
    end
end



# @testset "Testing norm invariance under permutations" begin
#     using Combinatorics
#     @testset "Trivial symmetries" begin
#         W = ℂ^2 ⊗ ℂ^3 ⊗ ℂ^4 ⊗ ℂ^5 ⊗ ℂ^6
#         t=Tensor(rand, W);
#         for k = 0:5
#             for p in permutations(1:5)
#                 p1 = ntuple(n->p[n],Val(k))
#                 p2 = ntuple(n->p[k+n],Val(5-k))
#                 t2 = permuteind(t,p1,p2)
#                 @test vecnorm(t2) ≈ vecnorm(t)
#             end
#         end
#     end
#     @testset "Abelian symmetries: ℤ₂ (self-dual)" begin
#         W = ℂ[ℤ₂](0=>1,1=>1) ⊗ ℂ[ℤ₂](0=>2,1=>1) ⊗ ℂ[ℤ₂](0=>3,1=>2) ⊗ ℂ[ℤ₂](0=>2,1=>3) ⊗ ℂ[ℤ₂](0=>1,1=>2)
#         t=Tensor(rand, W);
#         for k = 0:5
#             for p in permutations(1:5)
#                 p1 = ntuple(n->p[n], Val(k))
#                 p2 = ntuple(n->p[k+n], Val(5-k))
#                 t2 = permuteind(t, p1, p2)
#                 @test vecnorm(t2) ≈ vecnorm(t)
#             end
#         end
#     end
#     @testset "Abelian symmetries: ℤ₃ (not self-dual)" begin
#         W = ℂ[ℤ₃](0=>1,1=>1,2=>2) ⊗ ℂ[ℤ₃](0=>2,1=>1,2=>3) ⊗ ℂ[ℤ₃](0=>3,1=>2,2=>1) ⊗ ℂ[ℤ₃](0=>2,1=>3,2=>1) ⊗ ℂ[ℤ₃](0=>1,1=>2,2=>3)
#         t=Tensor(rand, W);
#         for k = 0:5
#             for p in permutations(1:5)
#                 p1 = ntuple(n->p[n],Val(k))
#                 p2 = ntuple(n->p[k+n],Val(5-k))
#                 t2 = permuteind(t,p1,p2)
#                 @test vecnorm(t2) ≈ vecnorm(t)
#             end
#         end
#     end
#     @testset "Abelian symmetries: U₁ (uses RepresentationSpace)" begin
#         W = ℂ[U₁](0=>1,1=>1,-1=>2) ⊗ ℂ[U₁](0=>2,1=>1,-1=>3) ⊗ ℂ[U₁](0=>3,1=>2,-1=>1) ⊗ ℂ[U₁](0=>2,1=>3,-1=>1) ⊗ ℂ[U₁](0=>1,1=>2,-1=>3)
#         t=Tensor(rand, W);
#         for k = 0:5
#             for p in permutations(1:5)
#                 p1 = ntuple(n->p[n],Val(k))
#                 p2 = ntuple(n->p[k+n],Val(5-k))
#                 t2 = permuteind(t,p1,p2)
#                 @test vecnorm(t2) ≈ vecnorm(t)
#             end
#         end
#     end
#     @testset "NonAbelian symmetries: SU₂" begin
#         W = ℂ[SU₂](0=>1,1//2=>1,1=>2) ⊗ ℂ[SU₂](0=>2,1//2=>1,1=>3) ⊗ ℂ[SU₂](0=>3,1//2=>2,1=>1) ⊗ ℂ[SU₂](0=>2,1//2=>3,1=>1) ⊗ ℂ[SU₂](0=>1,1//2=>2,1=>3)
#         t=Tensor(rand, W);
#         for k = 0:5
#             for p in permutations(1:5)
#                 p1 = ntuple(n->p[n],Val(k))
#                 p2 = ntuple(n->p[k+n],Val(5-k))
#                 t2 = permuteind(t,p1,p2)
#                 @test vecnorm(t2) ≈ vecnorm(t)
#             end
#         end
#     end
# end
