# Random-Features Hopfield Model 

Code needed to reproduce the results of the paper
_"Storage and Learning phase transitions in the Random-Features Hopfield Model"_ [[arxiv](https://arxiv.org/abs/2303.16880)]

This project contains three folders.
- `src` contains the code to run a zero-temperature asynchronous dynamics on a Random-Features Hopfeld Model and measure magnetisations. 
- `run` contains scripts to run the dynamics and reproduce the numerical results of the manuscript. Ready-to-plot results are provided one meaningful value of the parameters, along with the plot that compares them to the theoretical curve. 
- `saddle_point` contains the codes to solve the saddle point equations described in the manuscript.

## Numerical Results

### Example 
Below we show an example of how to generate a factor magnetisation curve
as a function of $\alpha$, given $N$ and $\alpha_d$
```julia
julia> include("RFHopfieldModel.jl"); RFH=RFHopfieldModel
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
           push!(res, [alpha q])
       end
julia> r = vcat(res...)
julia> plot(r[:,1], r[:,2]) # plot feature magnetisation as a function of alpha
```

### Reproduce results

The folder `run` contains script to produce a basic plot. To rerun the dynamic and reproduce the plot, type

```julia
julia> include("run_magnetisations.jl"); 
```

then, in the terminal, 

    python plot_magnetisations.py

which will produce `magnetisations_aD0.01.pdf` using the data files `magnetisations_aD0.01.txt` and `../saddle_point/factor/results/span_aD0.01.txt`. 

## Saddle-point solver

There is one folder for the equations of the leaning phase, called "factor", and one folder for the storage phase, called "pattern". Their structure is the same and is explained below.

The file `hopfield_RF_features_T=0.jl` contains the definition of the saddle-point equations and the essential functions to solve them, in a module called
`P`. The functions relevant to the user are the following: 
- `converge(...)` takes in input values $\alpha$, $\alpha_D$, and an initial condition of the order parameters and finds a fixed point by iterating the saddle-point equations.
- `span(...)` runs `converge(...)` for a given interval of $\alpha$ or $\alpha_D$, using the fixed point at the previous run as the initial condition of the next one. It prints all the fixed points on a file.
- `find_alphac(...)` runs `converge(...)` for a given interval of $\alpha$ or $\alpha_D$ and prints to a file only when the magnetisation changes abruptly, signaling the phase transition.

The scripts are designed to be used withing the julia REPL. 
An example of usage is 
```julia
julia> include("hopfield_RF_pattern_T=0.jl");
julia> P.converge(α=0.001,αD=1.00, verb=1,
            σ=:sign,resfile="results/alphac.txt", ψ=0.9, 
            dq=8.595353998742104e-9,qh=1.004060267804979,
            dp=1.2894207365173853e-7,dph=0.6366201987745321,
            pd=0.9999999929773609,dphd=0.6366205912296964,
            m=1.0
    )
```

The folders include basic scripts to run `span(...)` and  `find_alphac(...)`. They should be run withing the REPL after including the basic module:

```julia
julia> include("hopfield_RF_pattern_T=0.jl");
julia> include("run_find_alphac.jl")
```
In order to reproduce the phase diagram it is enough to plot the first two columns of the files produced by `find_alphac(...)`. We include the file we used in the folder: `alphac_factor.txt` and `alphac_pattern.txt`.

