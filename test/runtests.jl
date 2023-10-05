using Test, Random, TensorKit

include("choosetests.jl")

(; tests, sectors, exit_on_error, use_revise, seed) = choosetests(ARGS)

module Utility
include("utility.jl")
end
@eval(Utility, const sectorlist = eval.(Meta.parse.($sectors)))

for t in tests
    modname = Symbol("Test_($t)")
    m = @eval(Main, module $modname end)
    @eval(m, using Test, TestExtras, Random, TensorKit, ..Utility)
    !isnothing(seed) && Random.seed!(seed)
    @testset "$t" verbose = true begin
        Base.include(m, "$t.jl")
    end
end
