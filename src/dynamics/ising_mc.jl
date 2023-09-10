#
# MCMC on SK-like Ising model
# the couplings matrix J is assumed to be symmetric
#

function mc(J; s0=nothing,
               sweeps=100, verbose=false,
               seed=-1, β=0, dβ=0)

    seed > 0 && Random.seed!(seed)
    N = size(J, 1) 
    if s0 !== nothing
        @assert length(s0) == N
        s = copy(s0)
    else
        s = rand([-1f0,1f0], N)
    end
    
    conv = false
    it = 0
    h = zeros(Float32, N)
    E = energy(J, s)
    for _ = 1:sweeps
        count = 0
        for i = randperm(N)
            h[i] = @views dot(J[:,i], s)
            s[i] == signb(h[i]) && (count += 1)
            ΔE = 2 * h[i] * s[i]
            if ΔE <= 0 || (rand() < exp(-β * ΔE))
                s[i] *= -1
                E += ΔE/N 
            end
        end
        it += 1
        conv = (count == N)
        out = @sprintf("it=%i E=%.3f β=%.2E qt=%.2f", 
                        it, E, β, count/N)
        s0 !== nothing && (out *= @sprintf(" q0=%.3f", dot(s0,s)/N))
        verbose && print("\r"*out)
        β += dβ
        (β == Inf || dβ > 0) && conv && break
    end
    return conv, it, s
end
