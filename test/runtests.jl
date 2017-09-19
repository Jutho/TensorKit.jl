using TensorToolbox
using Base.Test


for k = 2:20
    p = randperm(k)
    swaps = permutation2swaps(p)
    r = collect(1:k)
    for i in swaps
        r[i], r[i+1] = r[i+1], r[i]
    end
    @test r == p
end
