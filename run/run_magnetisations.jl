include("../src/RFHopfieldModel.jl"); RFH=RFHopfieldModel

using LinearAlgebra, DelimitedFiles

alpha_d = 0.01
N = 10000

res = []

for alpha in LinRange(0.001, 0.3, 30)
    println("$(alpha)")

    ## pattern magnetisation
    s, J, x, z, C = RFH.run(N, alpha_d, alpha; 
                            qx=1.0, 
                            seed=21, seedx=23, 
                            binary_coeff=true, 
                            β=Inf, dβ=0.0, sweeps=200, 
                            verbose=true, more_info=false);

    m_pattern = dot(s, x[1]) / length(x[1])

    println()

    ## feature magnetisation
    s, J, x, z, C = RFH.run(N, alpha_d, alpha; 
                            qz=1.0, 
                            seed=21, seedx=23, 
                            binary_coeff=true, 
                            β=Inf, dβ=0.0, sweeps=200, 
                            verbose=true, more_info=false);

    m_feature = dot(s, z[1]) / length(z[1])

    println()

    push!(res, [alpha alpha_d m_pattern m_feature])
end

r = vcat(res...)

writedlm("magnetisations_aD$(alpha_d).txt", r)