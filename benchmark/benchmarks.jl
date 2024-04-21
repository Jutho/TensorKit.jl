params_path = get(ARGS, 1, joinpath("benchmark", "params.toml"))

using BenchmarkTools
using TensorKit
using TOML

include("utils.jl")
using .BenchmarkUtils

include("bench_linalg.jl")
include("bench_indexmanipulations.jl")
include("bench_tensornetworks.jl")

const SUITE = BenchmarkGroup()

# add some child groups
SUITE["linalg"] = BenchmarkGroup()
SUITE["indexmanipulations"] = BenchmarkGroup()
SUITE["tensornetworks"] = BenchmarkGroup()

# --------------------------------------------------------------------------------------- #
# load parameters
# --------------------------------------------------------------------------------------- #

all_parameters = TOML.parsefile(params_path)
for (group_id, group) in all_parameters
    haskey(SUITE, group_id) || addgroup!(SUITE, group_id)

    for (bench_id, params) in group
        @info "$group_id : $bench_id"
        f = eval(Meta.parse("benchmark_" * bench_id * "!"))
        f(SUITE[group_id], params)
    end
end
