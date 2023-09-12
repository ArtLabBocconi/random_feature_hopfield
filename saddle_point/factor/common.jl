import GSL: sf_log_erfc
using SpecialFunctions
using LinearAlgebra
#using Cxx
using Libdl
import Base.collect
using DelimitedFiles
using Parameters


# Importing shared library and header file
#const path_to_lib = pwd()
#addHeaderDir(path_to_lib, kind=C_System)
#Libdl.dlopen(path_to_lib * "/libOwens.so", Libdl.RTLD_GLOBAL)
#cxxinclude("owens.hpp")

using ExtractMacro
import ForwardDiff
import ForwardDiff: Dual, value, partials
# using Knet: minibatch
# using Plots; plotlyjs(size=(1000,800))

### SPECIAL FUNCTIONS ###

const F = Float64

∫d_(a,b,f) = quadgk(f, a, b, atol=1e-6, maxevals=10^7)[1]


G(x) = exp(-x^2/2) / F(√(2π))
H(x) = erfc(x /F(√2)) / 2
Hβ(x,β) = exp(-β) + (1-exp(-β))*H(x)
GH(x) = 2 / erfcx(x/F(√2)) / F(√(2π))
HG(x) =  F(√(2π))*erfcx(x/F(√2)) / 2
G(x, μ, Δ) = exp(-(x-μ)^2/(2Δ)) / √(2π*Δ)
logH(x) = sf_log_erfc(x/F(√2)) - log(F(2))
# logH(x) = x < -35.0 ? G(x) / x :
#           x >  35.0 ? -x^2 / 2 - log(2π) / 2 - log(x) :
#           log(H(x))

logHβ(x,β) = log(Hβ(x,β)) #no problems with the derivative seen
GHβ(x, β) = (1-exp(-β))*GH(x) / (exp(-β)/H(x) + (1-exp(-β)))

logG(x, μ, Δ) = -(x-μ)^2/(2Δ) - log(2π*Δ)/2
logG(x) = -x^2/2 - log(2π)/2

lrelu(x, γ=0.1f0) = max(x, γ*x)
log2cosh(x) = abs(x) + log1p(exp(-2abs(x)))
logcosh(x) = log2cosh(x) - log(2)

θfun(x) = x > 0 ? 1 : 0
crossentropy(x, γ) = log(1 + exp(-2γ*x))


# LOSS FUNCTIONS

square(x) = (1-x)^2 / 2
logistic(x) = log(1+exp(-x))
hinge(x) = max(0, 1-x)


abstract type AbstractParams end

### AUTOMATIC DIFFERENTIATION ###

grad = ForwardDiff.derivative
@inline function logH(d::Dual{T}) where T
    return Dual{T}(logH(value(d)), -GH(value(d)) * partials(d))
end

@inline function HG(d::Dual{T}) where T
    return Dual{T}(HG(value(d)), (value(d) * HG(value(d))-1) * partials(d))
end


argOwen(h,z) = exp(-h^2*(1+z^2)/2) / (2π*(1+z^2))
@inline function owenT(h, a::Dual{T}) where T
    #@show h, a
    return Dual{T}(owenT(h, value(a)), argOwen(h, value(a)) * partials(a))
end

@inline function owenT(h::Dual{T}, a) where T
    #@show h, a
    return Dual{T}(owenT(value(h), a), 0.5*G(value(h))*(1 - 2*H(-a*value(h))) * partials(h))
end

@inline function owenT(h::Dual{T}, a::Dual{T}) where T
    #@show h, a
    return Dual{T}(owenT(value(h), value(a)), 0.5*G(value(h))*(1 - 2*H(-value(a)*value(h))) * partials(h) + argOwen(value(h), value(a)) * partials(a))
end


### NUMERICAL DIFFERENTIATION ####

# Numerical Derivative
# Can take also directional derivative
# (tune the direction with i and δ).
function deriv(f::Function, i, x...; δ = 1e-5)
    x1 = deepcopy(x) |> collect
    x1[i] += δ #x1[i] .+= δ??????
    f0 = f(x...)
    f1 = f(x1...)
    return (f1-f0) / norm(δ)
end

# Numerical Derivative for member of the structured input
function deriv_(f::Function, i::Int, x...; arg=1, δ=1e-5)
    x1 = deepcopy(x)
    setfield!(x1[arg], i, getfield(x1[arg], i) + δ)
    f0 = f(x...)
    f1 = f(x1...)
    return (f1-f0) / δ
end



# x is to update; func is the function that updates x, and has params as arguments; Δ is the error; ψ the damping
macro update(x, func, Δ, ψ, verb, params...)
    name = string(x.args[2].value)
    # name = string(x.args[2].args[1]) # j0.6

    # if x isa Symbol || x.head == :ref
        # name = string(x.args[1], " ", eval(x.args[2]))
    # else
    #     name = string(x.args[2].args[1])
    # end
    x = esc(x)
    Δ = esc(Δ)
    ψ = esc(ψ)
    verb = esc(verb)
    params = map(esc, params)
    fcall = Expr(:call, func, params...)
    quote
        oldx = $x
        newx = $fcall
        abserr = norm(newx - oldx)     # p=2 by default, so |newx-oldx|
        relerr = abserr == 0 ? 0 : abserr / ((norm(newx) + norm(oldx)) / 2)
        $Δ = max($Δ, min(abserr, relerr))
        $x = (1 - $ψ) * newx + $ψ * oldx
        $verb > 1 && println("  ", $name, " = ", $x)
    end
end

macro updateI(x, ok, func, Δ, ψ, verb, params...)
    n = string(x.args[2].value)
    # n = string(x.args[2].args[1]) # j0.6
    x = esc(x)
    ok = esc(ok)
    Δ = esc(Δ)
    ψ = esc(ψ)
    verb = esc(verb)
    func = esc(func)
    params = map(esc, params)
    fcall = Expr(:call, func, params...)
    quote
        oldx = $x
        $ok, newx = $fcall
        if $ok
            abserr = abs(newx - oldx)
            relerr = abserr == 0 ? 0 : abserr / ((abs(newx) + abs(oldx)) / 2)
            $Δ = max($Δ, min(abserr, relerr))
            $x = (1 - $ψ) * newx + $ψ * oldx
            $verb > 1 && println("  ", $n, " = ", $x)
        else
            $verb > 1 && println("  ", $n, " = ", $x)
        end
    end
end


######## FILE PRINTING ######################

function exclusive(f::Function, fn::AbstractString = "lock.tmp")
    run(`lockfile -1 $fn`)
    try
        f()
    finally
        run(`rm -f $fn`)
    end
end

function shortshow(io::IO, x)
    T = typeof(x)
    print(io, T.name.name, "(", join([string(f, "=", getfield(x, f)) for f in fieldnames(T)], ","), ")")
end

function plainshow(x)
    T = typeof(x)
    join([getfield(x, f) for f in fieldnames(T)], " ")
end

function headershow(io::IO, T::Type, i0 = 0)
    print(io, join([string(i+i0,"=",f) for (i,f) in enumerate(fieldnames(T))], " "))
    return i0 + length(fieldnames(T))
end

function headershow(io::IO, x::String, i0 = 0)
    i0 += 1
    print(io, string(i0,"=",x," "))
    i0
end

function allheadersshow(io::IO, x...)
    i0 = 0
    print(io, "#")
    for y in x
        i0 = headershow(io, y, i0)
        print(io, " ")
    end
    println(io)
end

####### ROOT FINDING ##########

# using Roots: find_zero

# function findroot(f, x0; ftol=1e-8)
#     x = find_zero(f, x0, ftol=ftol)
#     return true, x
# end

using NLsolve: nlsolve, n_ary

function findroot(f, x0; ftol=1e-8)
    res = nlsolve(n_ary(f), [x0], ftol=ftol)
    true, res.zero[1]
end

function findroots(f!, x0; ftol=1e-8)
    res = nlsolve(f!, x0, ftol=ftol)
    res.zero[1], res.zero[2], res.zero[3]
end


# function findroot(f, x0; ftol=1e-8)
#     ok, x, it, normf0 = newton(f, x0, NewtonMethod(atol=ftol))
#     return ok, x
# end



#### NEWTON ####################

# authors: Carlo Baldassi and Carlo Lucibello
"""
    type NewtonMethod <: AbstractRootsMethod
        dx::Float64
        maxiters::Int
        verb::Int
        atol::Float64
    end
Type containg the parameters for Newton's root finding algorithm.
The default parameters are:
    NewtonMethod(dx=1e-7, maxiters=1000, verb=0, atol=1e-10)
"""
mutable struct NewtonMethod
    dx::Float64
    maxiters::Int
    verb::Int
    atol::Float64
end

mutable struct NewtonParameters
    δ::Float64
    ϵ::Float64
    verb::Int
    maxiters::Int
end

NewtonMethod(; dx=1e-7, maxiters=1000, verb=0, atol=1e-10) =
                                    NewtonMethod(dx, maxiters, verb, atol)

function ∇!(∂f::Matrix, f::Function, x0, δ, f0, x1)
    n = length(x0)
    copy!(x1, x0)
    for i = 1:n
        x1[i] += δ
        ∂f[:,i] = (f(x1) - f0) / δ
        x1[i] = x0[i]
    end
    #=cf = copy(∂f)=#
    #=@time ∂f[:,:] = @parallel hcat for i = 1:n
        x1[i] += δ
        d = (f(x1) - f0) / δ
        x1[i] = x0[i]
        d
    end=#
    #@assert cf == ∂f
end

∇(f::Function, x0::Real, δ::Real, f0::Real) = (f(x0 + δ) - f0) / δ

"""
    newton(f, x₀, pars=NewtonMethod())
Apply Newton's method with parameters `pars` to find a zero of `f` starting from the point
`x₀`.
The derivative of `f` is computed by numerical discretization. Multivariate
functions are supported.
Returns a tuple `(ok, x, it, normf)`.
**Usage Example**
ok, x, it, normf = newton(x->exp(x)-x^4, 1.)
ok || normf < 1e-10 || warn("Newton Failed")
"""
#note that in 1.0 warnings are eliminated at all
function newton(f, x₀::Float64, m=NewtonMethod())
    η = 1.0
    ∂f = 0.0
    x = x₀
    x1 = 0.0
    f0 = f(x)
    #@assert isa(f0, Real)
    normf0 = abs(f0)
    it = 0
    while normf0 ≥ m.atol
        #m.verb > 1 && println("normf0 = $normf0, maximum precision = $(m.atol)")
        it > m.maxiters && return (false, x, it, normf0)
        it += 1
        if m.verb > 1
            println("(𝔫) it=$it")
            println("(𝔫)   x=$x")
            println("(𝔫)   f(x)=$f0")
            println("(𝔫)   normf=$(abs(f0))")
            println("(𝔫)   η=$η")
        end
        δ = m.dx
        while true
            #∂f = ∇(f, x, δ, f0)
            try
                #∂f = grad(f, x)
                #m.verb > 1 && println("∇f = $(∂f)")
                ∂f = ∇(f, x, δ, f0)
                f1 = f(x + δ)
                #∂f_fd = grad(f, x)
                #m.verb > 1 && println("f1 = $(f1)\n ∇f = $(∂f) \t With FD: ∇f = $(∂f_fd) \t difference: Fd-num=$(∂f_fd-∂f)")
                m.verb > 1 && println("f1 = $(f1)\n ∇f = $(∂f)")
                break
            catch err
                #warn("newton: catched error:")
                #Base.display_error(err, catch_backtrace())
                δ /= 2
                #warn("new δ = $δ")
            end
            if δ < 1e-20
                #normf0 ≥ m.atol && warn("newton:  δ=$δ")
                println("Problema di δ!!")
                return (false, x, it, normf0)
            end
        end
        Δx = -f0 / ∂f
        m.verb > 1 && println("(𝔫)  Δx=$Δx")
        while true
            x1 = x + Δx * η
            local new_f0, new_normf0
            try
                new_f0 = f(x1)
                new_normf0 = abs(new_f0)
            catch err
                #warn("newton: catched error:")
                #Base.display_error(err, catch_backtrace())
                new_normf0 = Inf
            end
            if new_normf0 < normf0
                η = min(1.0, η * 1.1)
                f0 = new_f0
                normf0 = new_normf0
                x = x1
                break
            end
            # η is lowered if f(x1) fails, or if new_normf0 ≥ normf0
            η /= 2
            #η problem arises when the derivatives for example is ≈ 0 and x1 is really different from x and the new_normf0 ≫ normf0
            η < 1e-20 && println("Problema di η!!")
            η < 1e-20 && return (false, x, it, normf0)
        end
    end
    return true, x, it, normf0
end

function newton(f::Function, x₀, pars::NewtonParameters)
    η = 1.0
    n = length(x₀)
    ∂f = Array{Float64}(undef, n, n)
    x = Float64[x₀[i] for i = 1:n]  #order parameters
    x1 = Array{Float64}(undef, n)

    f0 = f(x)                       #system of equation
    @assert length(f0) == n
    @assert isa(f0, Union{Real,Vector})
    normf0 = norm(f0) #previous version and in the following: vecnorm--->norm
    it = 0
    while normf0 ≥ pars.ϵ
        it > pars.maxiters && return (false, x, it, normf0)
        it += 1
        if pars.verb > 1
            println("(𝔫) it=$it")
            println("(𝔫)   x=$x")
            println("(𝔫)   f0=$f0")
            println("(𝔫)   norm=$(norm(f0))")
            println("(𝔫)   η=$η")
        end
        δ = pars.δ
        while true
            try
                ∇!(∂f, f, x, δ, f0, x1)
                break
            catch
                δ /= 2
            end
            if δ < 1e-15
                #normf0 ≥ pars.ϵ && warn("newton:  δ=$δ")
                return (false, x, it, normf0)
            end
        end
        if isa(f0, Vector)
            Δx = -∂f \ f0
        else
            Δx = -f0 / ∂f[1,1]
        end
        pars.verb > 1 && println("(𝔫)  Δx=$Δx")
        while true
            for i = 1:n
                x1[i] = x[i] + Δx[i] * η
            end
            local new_f0, new_normf0
            try
                new_f0 = f(x1)
                new_normf0 = norm(new_f0)
            catch
                new_normf0 = Inf
            end
            if new_normf0 < normf0
                η = min(1.0, η * 1.1)
                if isa(f0, Vector)
                    copy!(f0, new_f0)
                else
                    f0 = new_f0
                end
                normf0 = new_normf0
                copy!(x, x1)
                break
            end
            η /= 2
            η < 1e-15 && return (false, x, it, normf0)
        end
    end
    return true, x, it, normf0
end
