#
# Utility functions for RFHopfieldModel.jl
#

signb(x::T) where {T} = x == zero(T) ? one(T) : sign(x)

function energy(J, s)
    return -dot(J * s, s) / (2*length(s))
end 

# given (s0, q) returns a conf. s with overlap q with s0
function close_conf(s0, q)
    N = length(s0)
    s = copy(s0)
    ndiff = round(Int, 0.5 * (1-q) * N)
    for i in randperm(N)[1:ndiff]
        s[i] *= -1
    end
    return s
end

# print info for retrieval exps 
function print_ret_info(s, x; nlines=10)
    m = [abs(dot(x[μ], s) / length(s)) for μ = 1:length(x)]
    idx_m = sortperm(m; rev=true)
    for i = 1:min(length(x), nlines)
        println("#$(idx_m[i]): \t q = $(m[idx_m[i]])")
    end
end

# Coupling Matrix 'J'
# assumes x is a vector of P vectors with length N
function create_couplings(x)
    N = length(first(x))
    X = hcat(x...)
    J = X * transpose(X) ./ N 
    J[diagind(J)] .= 0f0
    return J
end

# ±1 Patterns (iid) 
# P pattern of size N 
function create_patterns(N::Int, P::Int; seed=-1)
    seed > 0 && Random.seed!(seed)
    x = [rand([-1f0,1f0], N) for _ = 1:P]
    return x
end
function create_patterns(N::Int, α::T; 
                         seed=-1) where {T <: AbstractFloat}
    P = round(Int, N * α) # P can be even
    x = create_patterns(N, P; seed)
    return x
end

# ±1 Patterns (random features model) 
# P pattern of size N with hidden dimension D
function create_patterns(N::Int, D::Int, P::Int; 
                         binary_coeff=false, seed=-1)
    seed > 0 && Random.seed!(seed)
    Z = rand([-1f0, 1f0], D, N)
    C = randn(Float32, P, D)
    binary_coeff && (C = sign.(C))
    X = sign.(C * Z)
    x = @views [X[μ,:] for μ = 1:P]
    z = @views [Z[k,:] for k = 1:D]
    return x, z, C
end
function create_patterns(N::Int, d::Union{Int,T}, p::Union{Int,T}; 
                         binary_coeff=false, 
                         seed=-1) where {T <: AbstractFloat}

    P = (typeof(p) <: AbstractFloat ? round(Int, N * p) : p);  
    D = (typeof(d) <: AbstractFloat ? round(Int, N * d) : d);  
    binary_coeff && (D += iseven(D))
    x, z, C = create_patterns(N, D, P; binary_coeff, seed)
    return x, z, C
end 
