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

function ∫D_(a, f)
    a > ∞ && return 0.0
    int = interval[interval .> a]
    
    return quadgk(z->begin
        r = G(z) .* f(z)
        isfinite(r) ? r : 0.0
    end, a, int..., atol=1e-6, maxevals=10^5)[1]
end

############### PARAMS ################

@with_kw mutable struct OrderParams
	q::Float64 = 0.5
    qh::Float64 = 0.2
    t::Float64 = 0.1
    th::Float64 = 0.1
    m::Float64 = 0.3
    p::Float64
    ph::Float64
    pd::Float64
    phd::Float64
end

collect(op::OrderParams) = [getfield(op, f) for f in fieldnames(op)]

@with_kw mutable struct ExtParams
    β::Float64
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
    s::Float64 		# free energy
end


Base.show(io::IO, op::OrderParams) = shortshow(io, op)
Base.show(io::IO, ep::ExtParams) = shortshow(io, ep)
Base.show(io::IO, tf::ThermFunc) = shortshow(io, tf)

###################################################################################

#### INTERACTION AND ENTROPIC TERMS ################################
function Gi(dq, qh, dp, dph, pd, dphd, m, mh, α, αD)
    -(m*mh) - α/2 + (dp*dph*α)/2 - (pd*dphd*α)/2 + (dq*qh*α)/2 + (m^2*(-dph + dphd)*α/αD)/2
end
Gi(op::OrderParams, ep::ExtParams) = Gi(op.dq, op.qh, op.dp, op.dph, op.pd, op.dphd, op.m, op.mh, ep.α, ep.αD)


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

function Gs2(dq, dph, dphd, α, αD)
    term = 1 - (-dph + dphd)*(1 - dq)*α/αD
    -1/2*α*(-(((dph + 2*dph*dq - dphd*dq)*α/αD)/term) + log(term))/α/αD
end
Gs2(op::OrderParams, ep::ExtParams) = Gs2(op.dq, op.dph, op.dphd, ep.α, ep.αD)

#### CORRESPONDING UPDATES

function update_qh(q, t, p, ph, pd, phd, β, α, αD, σ)
    κ1, κ2, κs = compute_κ(σ)
    κss = κs^2
    κ1s = κ1^2
    αT = α/αD 

    αT*(t^2 * ph^2)/(αD*(1-β*αT*q*ph)^2) + (κss*(κss*q+κ1s*p))/(1-β*(κss*(1-q)+κ1s*(pd-p)))^2 +(ph + β*αT*q*(phd-ph)^2)/(β*(1-β*αT*(1-q)*(phd-ph))^2)
    
end
update_qh(op::OrderParams, ep::ExtParams) = update_qh(op.q, op.t, op.p, op.ph, op.pd, op.phd, ep.β, ep.α, ep.αD, ep.σ)

function update_th(q, t, ph, phd, β, α, αD)
    αT = α/αD

    αT*t*(phd/(1-β*αT*phd) - ph/(1-β*αT*q*ph))
end
update_th(op::OrderParams, ep::ExtParams) = update_th(op.q, op.t, op.ph, op.phd, ep.β, ep.α, ep.αD)

function update_ph(q, p, pd, β, σ)
    κ1, κ2, κs = compute_κ(σ)
    κss = κs^2
    κ1s = κ1^2
 
    β*κ1s*(κ1s*p + κss*q)/(1-β*(κ1s*(pd-p)+κss*(1-q)))^2
end
update_ph(op::OrderParams, ep::ExtParams) = update_ph(op.q, op.p, op.pd, ep.β, ep.σ)

function update_phd(q, p, pd, β, σ)
    κ1, κ2, κs = compute_κ(σ)
    κss = κs^2
    κ1s = κ1^2

    κ1s*(1-β*(κ1s*(pd-2*p)+κss*(1-2*q)))/(1-β*(κ1s*(pd-p)+κss*(1-q)))^2
end
update_phd(op::OrderParams, ep::ExtParams) = update_phd(op.q, op.p, op.pd, ep.β, ep.σ)

function update_q(qh, m, th, β, α)

    2*∫D_(0, v-> ∫D(x -> tanh(β*(m + th*v + x*√(α*qh-th^2)))^2))
end
update_q(op::OrderParams, ep::ExtParams) = update_q(op.qh, op.m, op.th, ep.β, ep.α)

function update_t(qh, m, th, β, α)
    2*β*m*(1-∫D(x -> tanh(β*x*√(α*qh-th^2))^2))/(√(2*π))
end
update_t(op::OrderParams, ep::ExtParams) = update_t(op.qh, op.m, op.th, ep.β, ep.α)

function update_m(qh, m, th, β, α)
    2*∫D_(0, v -> ∫D(x -> tanh(β*(m + v*th + x*√(α*qh-th^2)))))
end
update_m(op::OrderParams, ep::ExtParams) = update_m(op.qh, op.m, op.th, ep.β, ep.α)

function update_p(q, t, ph, phd, β, α, αD)
    αT = α/αD 

    (t^2)/(αD*(1-β*αT*q*ph)^2) + (q + β*αT*ph*(1-q)^2)/(1-β*αT*(1-q)*(phd-ph))^2
end
update_p(op::OrderParams, ep::ExtParams) = update_p(op.q, op.t, op.ph, op.phd, ep.β, ep.α,ep.αD)

function update_pd(q, t, ph, phd, β, α, αD)
    αT = α/αD 

    (t^2)/(αD*(1-β*αT*phd)^2) + (1 + β*αT*(2*ph-phd)*(1-q)^2)/(1-β*αT*(1-q)*(phd-ph))^2
end
update_pd(op::OrderParams, ep::ExtParams) = update_pd(op.q, op.t, op.ph, op.phd, ep.β, ep.α,ep.αD)


#### ENERGETIC TERM ################################################
# function Ge(qh, mh, α, β)
#     ∫D(z -> begin
#         term = √(α * qh) * z + mh
#         log(2 * cosh(β * term))
#     end) / β
# end
# Ge(op::OrderParams, ep::ExtParams) = Ge(op.qh, op.m, ep.α, ep.β)


############ Thermodynamic functions ############

function free_energy(op::OrderParams, ep::ExtParams)
    # Gi(op, ep) + Gs1(op, ep) + Gs2(op, ep) + Ge(op, ep)
    -99
end

## Thermodynamic functions
function all_therm_func(op::OrderParams, ep::ExtParams)
    s = entropy(op.q, op.qh, op.t, op.th, op.m, op.p, op.ph, op.pd, op.phd, ep.β, ep.α, ep.αD, ep.σ)
    return ThermFunc(s)
end

#################  SADDLE POINT  ##################
# Right-hand-side
fq(op, ep) = update_q(op, ep)			   	   # q = fq (der: qh)
fqh(op, ep) = update_qh(op, ep)	               # qh = fqh (der: dq)
ft(op, ep) = update_t(op, ep)	   		       # t = fth (der: dq)
fth(op, ep) = update_th(op, ep)	   		       # th = ft (der: dq)
fm(op, ep) = update_m(op, ep)			   	   # m = fm (der: m)
fp(op, ep) = update_p(op, ep)			   	  
fph(op, ep) = update_ph(op, ep)		   	   
fpd(op, ep) = update_pd(op, ep)			   	   
fphd(op, ep) = update_phd(op, ep)		   	   
	   	   


function converge!(op::OrderParams, ep::ExtParams, pars::Params)
    @extract pars: maxiters verb ϵ ψ
    Δ = Inf
    ok = false   	
    
    dth = 1e-7
    dqh = 1e-7

    for it = 1:maxiters
        Δ = 0.0
        ok = true
        verb > 1 && println("########## it=$it ##########")

        ########################################################################
       	
        @update  op.q    fq       Δ ψ verb  op ep   #update dq
        @update  op.qh   fqh      Δ ψ verb  op ep   #update qh
        diff = ep.α*op.qh - op.th^2
        fixed_diff = false
        if diff <= 0 
            verb > 1 && println("fixed_diff! $(diff)")
            op.qh = (op.th^2 + dqh)/(ep.α)
        end
        @update  op.t    ft       Δ ψ verb  op ep   #update t
        @update  op.th   fth      Δ ψ verb  op ep   #update th
        diff = ep.α*op.qh - op.th^2
        fixed_diff = false
        if diff <= 0 
            verb > 1 && println("fixed_diff! $(diff)")
            op.th = √(ep.α * op.qh - dth) 
        end
        @update  op.m    fm       Δ ψ verb  op ep   #update m
        @update  op.p    fp       Δ ψ verb  op ep   #update p
        @update  op.ph   fph      Δ ψ verb  op ep   #update ph
        @update  op.pd   fpd      Δ ψ verb  op ep   #update pd
        @update  op.phd  fphd     Δ ψ verb  op ep  #update qh

        ########################################################################

        verb > 1 && println(" Δ=$Δ\n")
        #(println(op); println(ep))
        #verb > 2 && it%5==0 && (println(ep);println(all_therm_func(op, ep));println(op))

        @assert isfinite(Δ)
        ok &= Δ < ϵ
        ok &= !fixed_diff
        ok && break         # if ok==true, exit
    end

    ok, Δ
end


function converge(;
    q = 0.5, qh = 0.2,
    t = 0.1, th = 0.1,
    m = 0.9, 
    p = 0.5, ph = 0.2, 
    pd = 0.5, phd = 0.2, 
    β = 2.0,
    α = 0.1, αD = 0.1, σ = :sign,
    ϵ = 1e-5, maxiters = 10000, verb = 3, ψ = 0.,
    kws...)

    op = OrderParams(q, qh, t, th, m, p, ph, pd, phd)
    ep = ExtParams(β, α, αD, σ)
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
    q = 0.5, qh = 0.2,
    t = 0.1, th = 0.1,
    m = 0.9, 
    p = 0.5, ph = 0.2, 
    pd = 0.5, phd = 0.2, 
    β = 2.0,
    α = 0.1, αD = 0.1, σ = :sign,
    ϵ = 1e-5, maxiters = 10000, verb = 3, ψ = 0.,
    kws...)

    op = OrderParams(q, qh, t, th, m, p, ph, pd, phd)
    ep = ExtParams(first(β), first(α), first(αD), σ)
    pars = Params(ϵ, ψ, maxiters, verb)

    return span!(op, ep, pars; β=β, α=α, αD=αD, kws...)
end

function span!(op::OrderParams, ep::ExtParams, pars::Params;
        β=2.0, α=0.2, αD=0.1, 
        resfile = "hopfield_RF_aD_patterns.txt")

    !isfile(resfile) && open(resfile, "w") do f
        allheadersshow(f, ExtParams, ThermFunc, OrderParams)
    end

    results = []

    for β in β, α in α, αD in αD
        ep.β = β;
        ep.α = α;
        ep.αD = αD;

        println("# NEW ITER: β=$(ep.β) α=$(ep.α) αD=$(ep.αD)")
        
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

function entropy(q, qh, t, th, m, p, ph, pd, phd, β, α, αD, σ)
    κ1, κ2, κs = compute_κ(σ)
    κss = κs^2
    κ1s = κ1^2
    αT = α/αD 

    Σ = κss*q + κ1s*p
    Σd = κss + κ1s*pd


    0.5*(t^2 * αT^2 * β^2 *((q*ph^2)/(1-β*αT*q*ph)^2 - (phd^2)/(1-β*αT*phd)^2) +
    + α*(1 + β*(β*qh*(1-q) +
    ((phd - ph)*(1-q)*(-1+β*αT*(phd - 2*q*phd+ph*(3*q-2))))/(1-β*αT*(phd - ph)*(1-q))^2 - (β*Σ*(Σd - Σ))/(1-β*(Σd-Σ))^2) - 1/(1-β*(Σd - Σ))) - α/αT * (log(1-β*αT*(1-q)*(phd-ph)) +
    αT*log(1-β*(Σd - Σ)))) +
    + 2*∫D_(0, v -> ∫D(x -> log(2*cosh(β*(m + v*th + x*√(α*qh-th^2))))))+
    - 2*β*∫D_(0, v -> ∫D(x -> (m + v*th + x*√(α*qh-th^2))* tanh(β*(m + v*th + x*√(α*qh-th^2)))))
end

entropy(op::OrderParams, ep::ExtParams) = entropy(op.q, op.qh, op.t, op.th, op.m, op.p, op.ph, op.pd, op.phd, ep.β, ep.α, ep.αD, ep.σ)
# function find_alphac(; 
#     dq = 0.5, qh = 0.2, 
#     dp = 0.5, dph = 0.2, 
#     pd = 0.5, dphd = 0.2, 
#     m = 0.3,
#     α = 0.1, αD = 1.0, σ = :sign,
#     ϵ = 1e-5, maxiters = 10000, verb = 3, ψ = 0.,
#     kws...)


#     op = OrderParams(dq, qh, dp, dph, pd, dphd, m)
#     ep = ExtParams(first(α), first(αD), σ)
#     pars = Params(ϵ, ψ, maxiters, verb)

#     return find_alphac!(op, ep, pars; αD=αD, α=α, kws...)
# end


# function find_alphac!(op::OrderParams, ep::ExtParams, pars::Params;
#     α=0.7, αD=0.5,
#     resfile = "alphac.txt")

#     !isfile(resfile) && open(resfile, "w") do f
#         allheadersshow(f, ExtParams, ThermFunc, OrderParams)
#     end

#     ok, Δ = converge!(op, ep, pars)

#     epn = deepcopy(ep)
#     opn = deepcopy(op)

#     for αD in αD, α in α, 
#     # for α in α, αD in αD
#         epn.α = α;
#         epn.αD = αD;

#         println("# NEW ITER: α=$(epn.α)  αD=$(epn.αD)")
        
# 	    ok, Δ = converge!(opn, epn, pars)

#         tf = all_therm_func(opn, epn)

#         # println(opn.m, " ",op.m, " ",abs(opn.m - op.m))
#         ok && abs(opn.m - op.m) > 0.3 && open(resfile, "a") do rf
#             println(rf, plainshow(epn), " ", plainshow(tf), " ", plainshow(opn))
#         end
#         # if (ok && abs(opn.m - op.m) > 0.3)
#         #     println(plainshow(epn), " ", plainshow(tf), " ", plainshow(opn))
#         # end

#         ep = deepcopy(epn)
#         op = deepcopy(opn)

#         !ok && break
#         pars.verb > 0 && print(ep, "\n", tf,"\n")
#     end

# end

end #module
