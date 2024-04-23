"""
    precompile_sector(I::Type{<:Sector})

Precompile common methods for the given sector type.
"""
function precompile_sector(::Type{I}) where {I<:Sector}
    precompile(Nsymbol, (I, I, I))
    precompile(Fsymbol, (I, I, I, I, I, I))
    precompile(Rsymbol, (I, I, I))
    precompile(Asymbol, (I, I, I))
    precompile(Bsymbol, (I, I, I))

    precompile(⊗, (I,))
    precompile(⊗, (I, I))
    precompile(⊗, (I, I, I))
    precompile(⊗, (I, I, I, I))

    precompile(FusionStyle, (I,))
    precompile(BraidingStyle, (I,))

    precompile(dim, (I,))
    precompile(sqrtdim, (I,))
    precompile(isqrtdim, (I,))
    precompile(dual, (I,))
    precompile(twist, (I,))
    precompile(frobeniusschur, (I,))

    try
        precompile(fusiontensor, (I, I, I))
    catch
    end
end
