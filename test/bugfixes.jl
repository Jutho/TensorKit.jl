@timedtestset "Bugfixes" verbose = true begin
    @testset "BugfixConvert" begin
        v = TensorMap(randn, ComplexF64,
                      (Vect[(Irrep[U₁] ⊠ Irrep[SU₂] ⊠ FermionParity)]((-3, 1 / 2, 1) => 3,
                                                                      (-5, 1 / 2, 1) => 10,
                                                                      (-7, 1 / 2, 1) => 13,
                                                                      (-9, 1 / 2, 1) => 9,
                                                                      (-11, 1 / 2, 1) => 1,
                                                                      (-5, 3 / 2, 1) => 3,
                                                                      (-7, 3 / 2, 1) => 3,
                                                                      (-9, 3 / 2, 1) => 1) ⊗
                       Vect[(Irrep[U₁] ⊠ Irrep[SU₂] ⊠ FermionParity)]((1, 1 / 2, 1) => 1)') ←
                      Vect[(Irrep[U₁] ⊠ Irrep[SU₂] ⊠ FermionParity)]((-3, 1 / 2, 1) => 3,
                                                                     (-5, 1 / 2, 1) => 10,
                                                                     (-7, 1 / 2, 1) => 13,
                                                                     (-9, 1 / 2, 1) => 9,
                                                                     (-11, 1 / 2, 1) => 1,
                                                                     (-5, 3 / 2, 1) => 3,
                                                                     (-7, 3 / 2, 1) => 3,
                                                                     (-9, 3 / 2, 1) => 1))
        w = convert(typeof(real(v)), v)
        @test w == v
        @test scalartype(w) == Float64
    end
end
