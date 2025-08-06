using TensorKit
using BenchmarkTools
using LinearAlgebra

LinearAlgebra.BLAS.set_num_threads(1)
TensorKit.Strided.set_num_threads(1)

verbose = true

include("TensorKitBenchmarks/TensorKitBenchmarks.jl")

suite = TensorKitBenchmarks.load!("tensornetworks")
suite = suite["tensornetworks"]["mpo"]

TensorKit.set_num_transformer_threads(1)
result_1 = BenchmarkTools.run(suite; verbose)
BenchmarkTools.save(joinpath(@__DIR__, "result_1.json"), result_1)

TensorKit.set_num_transformer_threads(2)
result_2 = BenchmarkTools.run(suite; verbose)
BenchmarkTools.save(joinpath(@__DIR__, "result_2.json"), result_2)

TensorKit.set_num_transformer_threads(4)
result_4 = BenchmarkTools.run(suite; verbose)
BenchmarkTools.save(joinpath(@__DIR__, "result_4.json"), result_2)

estimator = minimum
judgement1 = collect(judge(estimator(result_2), estimator(result_1)))
judgement2 = collect(judge(estimator(result_4), estimator(result_1)))

sort!(judgement1; by=((x, y),) -> (x[1], x[2], x[3]))
sort!(judgement2; by=((x, y),) -> (x[1], x[2], x[3]))
