# Random-Features Hopfield Model 

Code needed to reproduce the results of the pre-print  
_"Storage and Learning phase transitions in the Random-Features Hopfield Model"_ [[arxiv](https://arxiv.org/abs/2303.16880)]

## General Usage 
Todo...
### Example 
Below we show an example of how to generate a factor magnetization curve
as a function of $\alpha$, given $N$ and $\alpha_d$
```julia
julia> include("RFHopfiledModel.jl); RFH=RFHopfieldModel
julia> using LinearAlgebra, PyPlot
julia> alpha_d = 0.05; N = 4000
julia> res = []
julia> for alpha in LinRange(0.1, 0.6, 20)
           s, J, x, z, C = RFH.run(N, alpha_d, alpha; 
                                   qz=1.0, 
                                   seed=21, seedx=23, 
                                   binary_coeff=true, 
                                   β=Inf, dβ=0.0, sweeps=200, 
                                   verbose=false, more_info=false);
           q = dot(s, z[1]) / length(z[1])
           push!(res, [a q])
       end
julia> r = vcat(res...)
julia> plot(r[:,1], r[:,2]) # plot feature magnetization as a function of alpha
```
