module P

using QuadGK
using ForwardDiff
using Optim

include("common.jl")


###### INTEGRATION  ######
const ∞ = 10.0
const dx = 0.01

const interval = map(x->sign(x)*abs(x)^2, -1:dx:1) .* ∞

∫D(f, int=interval) = quadgk(z->begin
        r = G(z) .* f(z)
        isfinite(r) ? r : 0.0
    end, int..., atol=1e-7, maxevals=10^7)[1]


############### PARAMS ################

@with_kw mutable struct OrderParams
	dq::Float64 = 0.5
    qh::Float64 = 0.2
	dp::Float64 = 0.5
    dph::Float64 = 0.2
    pd::Float64 = 0.5
    dpdh::Float64 = 0.2
    m::Float64 = 0.3
    mh::Float64 = 0.3
end

collect(op::OrderParams) = [getfield(op, f) for f in fieldnames(op)]

@with_kw mutable struct ExtParams
    α::Float64 = 0.1  # constrained density
    αD::Float64 = 0.1  # constrained density
    σ::Symbol = :sign # non-linearity of the projection
end

@with_kw mutable struct Params
    ϵ::Float64 = 1e-5       # stop criterium
    ψ::Float64 = 0.         # damping
    maxiters::Int = 10000
    verb::Int = 2
end

mutable struct ThermFunc
    f::Float64 		# free energy
end

collect(tf::ThermFunc) = [getfield(tf, f) for f in fieldnames(tf)]

Base.show(io::IO, op::OrderParams) = shortshow(io, op)
Base.show(io::IO, ep::ExtParams) = shortshow(io, ep)
Base.show(io::IO, tf::ThermFunc) = shortshow(io, tf)

###################################################################################

#### INTERACTION AND ENTROPIC TERMS ################################
function Gi(dq, qh, dp, dph, pd, dpdh, m, mh, α, αD)
    -(m*mh) - α/2 + (dp*dph*α)/2 - (pd*dpdh*α)/2 + (dq*qh*α)/2 + (m^2*(-dph + dpdh)*α/αD)/2
end
Gi(op::OrderParams, ep::ExtParams) = Gi(op.dq, op.qh, op.dp, op.dph, op.pd, op.dpdh, op.m, op.mh, ep.α, ep.αD)


function compute_κ(σ)
	if σ == :sign
		κ1 = √(2/π) # 0.5
		κ2 = 1.0
	elseif σ == :id
        κ1 = 1
        κ2 = 1
    else
		κ1 = @eval ∫D(z -> z*$σ(z))
		κ2 = @eval ∫D(z -> $σ(z)^2)
	end

	κs = √(κ2 - κ1^2)
	return κ1, κ2, κs
end

function Gs1(dq, dp, pd, α, σ)
    κ1, κ2, κs = compute_κ(σ)
    kss = κs^2
    k1s = κ1^2

    term = 1 - (kss - k1s*dp + k1s*pd - kss*dq)*β
    -1/2*α*(-(((k1s*dp + kss*dq)*β)/term) + log(term))
end
Gs1(op::OrderParams, ep::ExtParams) = Gs1(op.dq, op.dp, op.pd, ep.α, ep.σ)

function Gs2(dq, dph, dpdh, α, αD)
    term = 1 - (-dph + dpdh)*(1 - dq)*α/αD
    -1/2*α*(-(((dph + 2*dph*dq - dpdh*dq)*α/αD)/term) + log(term))/α/αD
end
Gs2(op::OrderParams, ep::ExtParams) = Gs2(op.dq, op.dph, op.dpdh, ep.α, ep.αD)

#### CORRESPONDING UPDATES

function update_qh(dph, dpdh, dq, dp, pd, α, αD, σ)
    κ1, κ2, κs = compute_κ(σ)
    kss = κs^2
    k1s = κ1^2

    kss*(k1s*pd+kss)/(1-k1s*dp -kss*dq)^2 + (dpdh+α/αD*dph^2)/(1-α/αD*dq*dph)^2
end
update_qh(op::OrderParams, ep::ExtParams) = update_qh(op.dph, op.dpdh, op.dq, op.dp, op.pd, ep.α, ep.αD, ep.σ)

function update_dph(dp,dq,σ)
    κ1, κ2, κs = compute_κ(σ)
    kss = κs^2
    k1s = κ1^2

    k1s / (1-k1s*dp-kss*dq)
end
update_dph(op::OrderParams, ep::ExtParams) = update_dph(op.dp,op.dq,ep.σ)

function update_dpdh(dp,dq,pd,σ)
    κ1, κ2, κs = compute_κ(σ)
    kss = κs^2
    k1s = κ1^2

    k1s*(k1s*pd+kss)/(1-k1s*dp-kss*dq)
end
update_dpdh(op::OrderParams, ep::ExtParams) = update_dpdh(op.dp,op.dq,op.pd,ep.σ)

function update_dp(dq,dph,α,αD)
    dq/(1 - α/αD * dq * dph)
end
update_dp(op::OrderParams, ep::ExtParams) = update_dp(op.dq,op.dph,ep.α,ep.αD)

function update_pd(dph,dpdh,dq,m,α,αD)
    m^2/αD + (1+α/αD*dq^2*dpdh)/(1-α/αD*dq*dph)^2
end
update_pd(op::OrderParams, ep::ExtParams) = update_pd(op.dph,op.dpdh,op.dq,op.m,ep.α,ep.αD)

function update_mh(dph,m,α,αD)
    α/αD*dph*m
end
update_mh(op::OrderParams, ep::ExtParams) = update_mh(op.dph,op.m,ep.α,ep.αD)

#### ENERGETIC TERM ################################################
function Ge(qh, mh, α, β)
    ∫D(z -> begin
        term = √(α * qh) * z + mh
        log(2 * cosh(β * term))
    end) / β
end
Ge(op::OrderParams, ep::ExtParams) = Ge(op.qh, op.mh, ep.α, ep.β)

#### CORRESPONDING UPDATES

function update_dq(qh, mh, α)
    term =  1 / √(α * qh)
    2 * G(-mh * term) * term
end
update_dq(op::OrderParams, ep::ExtParams) = update_dq(op.qh, op.mh, ep.α)

function update_m(qh, mh, α)
    term =  1 / √(α * qh)
    2 * H(- mh * term) - 1
end
update_m(op::OrderParams, ep::ExtParams) = update_m(op.qh, op.mh, ep.α)

############ Thermodynamic functions ############
    

function free_energy(dq, qh, dp, dph, pd, dpdh, m, mh, α, αD, σ)
    κ1, κ2, κs = compute_κ(σ)
    kss = κs^2
    k1s = κ1^2

    αT = α/αD

    0.5*α*(1 + dp*dpdh + dph*pd + dq*qh) - 0.5*α*((kss + k1s*pd)/(1- k1s*dp - kss*dq) + (dph + dq*dpdh)/(1-dph*dq*αT)) - 0.5*αT*dph*m^2 + m*mh +
    -0.5*∫D(z -> ((z*√(α*qh) + mh)*(2*θfun(z*√(α*qh) + mh)-1)))
end

free_energy(op::OrderParams, ep::ExtParams) = free_energy(op.dq, op.qh, op.dp, op.dph, op.pd, op.dpdh, op.m, op.mh, ep.α, ep.αD, ep.σ)

## Thermodynamic functions
function all_therm_func(op::OrderParams, ep::ExtParams)
    # f = free_energy(op.dq, op.qh, op.dp, op.dph, op.pd, op.dpdh, op.m, op.mh, ep.α, ep.αD, ep.σ)
    f = -99
    return ThermFunc(f)
end

#################  SADDLE POINT  ##################
# Right-hand-side
fq(op, ep) = update_dq(op, ep)			   	   # dq = fq (der: qh)
fqh(op, ep) = update_qh(op, ep)		   		   # qh = fqh (der: dq)
fp(op, ep) = update_dp(op, ep)			   	   # dq = fq (der: qh)
fph(op, ep) = update_dph(op, ep)		   	   # qh = fqh (der: dq)
fpd(op, ep) = update_pd(op, ep)			   	   # dq = fq (der: qh)
fpdh(op, ep) = update_dpdh(op, ep)		   	   # qh = fqh (der: dq)
fm(op, ep) = update_m(op, ep)			   	   # m = fm (der: m)
fmh(op, ep) = update_mh(op, ep)			   	   # m = fm (der: m)


function converge!(op::OrderParams, ep::ExtParams, pars::Params)
    @extract pars: maxiters verb ϵ ψ
    Δ = Inf
    ok = false   	

    for it = 1:maxiters
        Δ = 0.0
        ok = true
        verb > 1 && println("########## it=$it ##########")

        ########################################################################
       	
        @update  op.dq    fq       Δ ψ verb  op ep   #update dq
        @update  op.qh   fqh      Δ ψ verb  op ep  #update qh
        @update  op.dp    fp       Δ ψ verb  op ep   #update dq
        @update  op.dph   fph      Δ ψ verb  op ep  #update qh
        @update  op.pd   fpd      Δ ψ verb  op ep   #update dq
        @update  op.dpdh  fpdh     Δ ψ verb  op ep  #update qh
        @update  op.m    fm       Δ ψ verb  op ep   #update m
        @update  op.mh   fmh      Δ ψ verb  op ep   #update m

        ########################################################################

        verb > 1 && println(" Δ=$Δ\n")
        (println(op); println(ep))
        #verb > 2 && it%5==0 && (println(ep);println(all_therm_func(op, ep));println(op))

        @assert isfinite(Δ)
        ok &= Δ < ϵ
        ok && break         # if ok==true, exit
    end

    ok, Δ
end


function converge(;
    dq = 0.5, qh = 0.2, 
    dp = 0.5, dph = 0.2, 
    pd = 0.5, dpdh = 0.2, 
    m = 0.3, mh = 0.3,
    α = 0.1, αD = 0.1, σ = :sign,
    ϵ = 1e-5, maxiters = 10000, verb = 3, ψ = 0.,
    kws...)

    op = OrderParams(dq,qh,dp,dph,pd,dpdh,m,mh)
    ep = ExtParams(α,αD,σ)
    pars = Params(ϵ, ψ, maxiters, verb)
	
    converge!(op, ep, pars)
    tf = all_therm_func(op, ep)
    println(tf)
    return op, ep, pars, tf
end



"""
    readparams(file::String, line::Int=-1)

Read order and external params from results file.
Zero or negative line numbers are counted
from the end of the file.
"""
function readparams(file::String, line::Int=0)
    data = readdlm(file, String)
    l = line > 0 ? line : length(data[:,2]) + line
    v = map(x-> begin
                    try
                        parse(Float64, x)
                    catch
                        Symbol(x)
                    end
                end, data[l,:])
    #return v
    i0 = length(fieldnames(ExtParams))
    i1 = i0 + 1  + length(fieldnames(ThermFunc))
    iend = i1 - 1 + length(fieldnames(OrderParams))

    return ExtParams(v[1:i0]...), OrderParams(v[i1:iend]...)
end


function span(;
    dq = 0.5, qh = 0.2, 
    dp = 0.5, dph = 0.2, 
    pd = 0.5, dpdh = 0.2, 
    m = 0.3, mh = 0.3,
    α = 0.1, αD = 0.1, σ = :sign,
    ϵ = 1e-5, maxiters = 10000, verb = 3, ψ = 0.,
    kws...)

    op = OrderParams(dq, qh, dp, dph, pd, dpdh, m, mh)
    ep = ExtParams(first(α), first(αD), σ)
    pars = Params(ϵ, ψ, maxiters, verb)

    return span!(op, ep, pars; α=α, αD=αD, kws...)
end

function span!(op::OrderParams, ep::ExtParams, pars::Params;
        α=0.2, αD=0.1, 
        resfile = "hopfield_RF_aD_T=0.txt")

    !isfile(resfile) && open(resfile, "w") do f
        allheadersshow(f, ExtParams, ThermFunc, OrderParams)
    end

    results = []

    for α in α, αD in αD
        ep.α = α;
        ep.αD = αD;

        println("# NEW ITER: α=$(ep.α) αD=$(ep.αD)")
        
	    ok, Δ = converge!(op, ep, pars)

        tf = all_therm_func(op, ep)

        push!(results, (ok, deepcopy(ep),  deepcopy(tf), deepcopy(op)))

        ok && open(resfile, "a") do rf
            println(rf, plainshow(ep), " ", plainshow(tf), " ", plainshow(op))
        end

        !ok && break
        pars.verb > 0 && print(ep, "\n", tf,"\n")
    end
    return results
end




function span_fixαT(;
    dq = 0.5, qh = 0.2, 
    dp = 0.5, dph = 0.2, 
    pd = 0.5, dpdh = 0.2, 
    m = 0.3, mh = 0.3,
    α = 0.1, αT = 1.0, σ = :sign,
    ϵ = 1e-5, maxiters = 10000, verb = 3, ψ = 0.,
    kws...)

    αD = first(α) / first(αT)

    op = OrderParams(dq, qh, dp, dph, pd, dpdh, m, mh)
    ep = ExtParams(first(α), αD, σ)
    pars = Params(ϵ, ψ, maxiters, verb)

    return span_fixαT!(op, ep, pars; αT=αT, α=α, kws...)

end


function span_fixαT!(op::OrderParams, ep::ExtParams, pars::Params;
    α=0.7, αT=0.5,
    resfile = "hopfield_RF_aD_T=0.txt")

    !isfile(resfile) && open(resfile, "w") do f
        allheadersshow(f, ExtParams, ThermFunc, OrderParams)
    end

    results = []

    for α in α, αT in αT
        ep.α = α;
        # ep.αD = αD;
        ep.αD = ep.α / αT


        println("# NEW ITER: α=$(ep.α) αD=$(ep.αD)  αT=$(αT)")
        
	    ok, Δ = converge!(op, ep, pars)

        tf = all_therm_func(op, ep)

        push!(results, (ok, deepcopy(ep), deepcopy(tf), deepcopy(op)))

        ok && open(resfile, "a") do rf
            println(rf, plainshow(ep), " ", plainshow(tf), " ", plainshow(op))
        end

        !ok && break
        pars.verb > 0 && print(ep, "\n", tf,"\n")
    end
    return results

end



function find_alphac(; 
    dq = 0.5, qh = 0.2, 
    dp = 0.5, dph = 0.2, 
    pd = 0.5, dpdh = 0.2, 
    m = 0.3, mh = 0.3,
    α = 0.1, αD = 1.0, σ = :sign,
    ϵ = 1e-5, maxiters = 10000, verb = 3, ψ = 0.,
    kws...)


    op = OrderParams(dq, qh, dp, dph, pd, dpdh, m, mh)
    ep = ExtParams(first(α), first(αD), σ)
    pars = Params(ϵ, ψ, maxiters, verb)

    return find_alphac!(op, ep, pars; αD=αD, α=α, kws...)
end


function find_alphac!(op::OrderParams, ep::ExtParams, pars::Params;
    α=0.7, αD=0.5,
    resfile = "alphac.txt")

    !isfile(resfile) && open(resfile, "w") do f
        allheadersshow(f, ExtParams, ThermFunc, OrderParams)
    end

    ok, Δ = converge!(op, ep, pars)

    epn = deepcopy(ep)
    opn = deepcopy(op)

    for α in α, αD in αD
        epn.α = α;
        epn.αD = αD;

        println("# NEW ITER: α=$(epn.α)  αD=$(epn.αD)")
        
	    ok, Δ = converge!(opn, epn, pars)

        tf = all_therm_func(opn, epn)

        println(opn.m - op.m)
        ok && abs(opn.m - op.m) > 0.3 && open(resfile, "a") do rf
        # ok && abs(opn.m - op.m) > 0.3 && ep.α == epn.α && open(resfile, "a") do rf
            println(rf, plainshow(epn), " ", plainshow(tf), " ", plainshow(opn))
        end

        ep = deepcopy(epn)
        op = deepcopy(opn)

        !ok && break
        pars.verb > 0 && print(ep, "\n", tf,"\n")
    end

end

end #module
