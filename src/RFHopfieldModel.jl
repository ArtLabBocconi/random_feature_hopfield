module RFHopfieldModel

using Statistics, Random, LinearAlgebra
using Printf, DelimitedFiles

include("utils/utils.jl")
include("dynamics/ising_mc.jl")

# d and p can be integers (interpreted as D, P)
# or floats (interpreted as αd, α)
# seed is for the dynamics and seedx is for the patterns
function run(N::Int, d, p; qx=0, qz=0,  
             seed=23, seedx=23, binary_coeff=false,
             # mc kws
             β=Inf, dβ=0, sweeps=100,
             verbose=false, more_info=true, save_conf=false, 
             conf_file="tmp.dat", exp_folder="../data/conf/")
    
    save_conf && @assert !isempty(conf_file) 
    
    # create patterns and couplings 
    if d > 0 
        x, z, C = create_patterns(N, d, p; binary_coeff, seed=seedx)
    else 
        x = create_patterns(N, p; seed=seedx)
    end 
    J = create_couplings(x)
  
    # init spin conf 
    if qx > 0 
        # init with overlap qx with first pattern 
        @assert qz <= 0
        s0 = close_conf(x[1], qx)
    elseif qz > 0 
        # init with overlap qz with first feature
        @assert qx == 0; @assert d > 0 
        s0 = close_conf(z[1], qz)
    else 
        # random init
        @assert qx == 0 && qz <= 0
        s0 = rand([-1f0, 1f0], N)
    end
    
    # run dynamics 
    conv, it, s =  mc(J; seed, s0, sweeps, verbose, β, dβ)

    # print more info about patterns/features mag.
    if more_info
        println("\n\nMost correlated patterns:")
        print_ret_info(s, x)
        if d > 0 
            println("\nMost correlated features:")
            print_ret_info(s, z)
        end 
    end

    # save final spin conf.
    if save_conf
        !ispath(exp_folder) && mkpath(exp_folder) 
        @info "Writing conf in $(exp_folder * conf_file)..."
        writedlm(exp_folder * conf_file, s)
    end 

    d > 0 && return s, J, x, z, C
    return s, J, x
end

end # module
