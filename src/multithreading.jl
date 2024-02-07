# global variables to control multi-threading behaviors 
global nthreads_mul::Int64
global nthreads_eigh::Int64
global nthreads_svd::Int64
global nthreads_add::Int64

function set_num_threads_mul(n::Int64)
     @assert 1 ≤ n ≤ Threads.nthreads()
     global nthreads_mul = n
     return nothing
end
get_num_threads_mul() = nthreads_mul

function set_num_threads_add(n::Int64)
     @assert 1 ≤ n ≤ Threads.nthreads()
     global nthreads_add = n
     return nothing
end
get_num_threads_add() = nthreads_add

function set_num_threads_svd(n::Int64)
     @assert 1 ≤ n ≤ Threads.nthreads()
     global nthreads_svd = n
     return nothing
end
get_num_threads_svd() = nthreads_svd

function set_num_threads_eigh(n::Int64)
     @assert 1 ≤ n ≤ Threads.nthreads()
     global nthreads_eigh = n
     return nothing
end
get_num_threads_eigh() = nthreads_eigh
