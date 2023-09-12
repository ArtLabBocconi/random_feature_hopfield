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
	q::Float64 = 0.5
    qh::Float64 = 0.2
	p::Float64 = 0.5
    ph::Float64 = 0.2
    pd::Float64 = 0.5
    pdh::Float64 = 0.2
    m::Float64 = 0.3
    mh::Float64 = 0.3
end

collect(op::OrderParams) = [getfield(op, f) for f in fieldnames(op)]

@with_kw mutable struct ExtParams
    α::Float64 = 0.1  # constrained density
    αD::Float64 = 0.1  # constrained density
    β::Float64 = 1.0  # inverse temperature
    σ::Symbol = :sign # non-linearity of the projection
end

@with_kw mutable struct Params
    ϵ::Float64 = 1e-5       # stop criterium
    ψ::Float64 = 0.         # damping
    maxiters::Int = 10000
    verb::Int = 2
end

mutable struct ThermFunc
    e::Float64 		# energy
    s::Float64      # entropy
end

collect(tf::ThermFunc) = [getfield(tf, f) for f in fieldnames(tf)]

Base.show(io::IO, op::OrderParams) = shortshow(io, op)
Base.show(io::IO, ep::ExtParams) = shortshow(io, ep)
Base.show(io::IO, tf::ThermFunc) = shortshow(io, tf)

###################################################################################

#### INTERACTION AND ENTROPIC TERMS ################################
function Gi(q, qh, p, ph, pd, pdh, m, mh, α, αD)
    # check (q*qh*α)/2 TODO
    0.5*α*(1+ -β*qh*(q-1) + (pd*pdh)- ph*p - (pdh-ph)*m^2/αD) + m*mh
end
Gi(op::OrderParams, ep::ExtParams) = Gi(op.q, op.qh, op.p, op.ph, op.pd, op.pdh, op.m, op.mh, ep.α, ep.αD)


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

function Gs1(q, p, pd, α, β, σ)
    κ1, κ2, κs = compute_κ(σ)
    kss = κs^2
    k1s = κ1^2
    
    Σ = kss*q + k1s*p
    Σd = kss + k1s*pd 
    term = 1-β*(Σd - Σ)

    return α/(2*β)*(log(term) - β*Σ/(term))
end

Gs1(op::OrderParams, ep::ExtParams) = Gs1(op.q, op.p, op.pd, ep.α, ep.β, ep.σ)

function Gs2(q, ph, pdh, α, αD)
    # CHECK TODO β
    term = 1 - (α/αD)*β*(pdh -ph)*(1 - q)
    return 0.5 * (αD/β) * ( log(term) - (α/αD)*β*((ph + pdh*q - 2*q*ph))/term) 
end

Gs2(op::OrderParams, ep::ExtParams) = Gs2(op.q, op.ph, op.pdh, ep.α, ep.αD)

#### CORRESPONDING UPDATES

function update_qh(ph, pdh, q, p, pd, α, αD, β, σ)
    κ1, κ2, κs = compute_κ(σ)
    kss = κs^2
    k1s = κ1^2
    αT = α/αD

    term1 = kss * (k1s*p + kss*q) / (1 + β * k1s * (p - pd) + β * kss * (q-1))^2
    term2 = (ph + αT * β * q *(ph - pdh)^2) / (β*( αT * β * (ph - pdh)*(q-1) - 1)^2)

    term1 + term2
end
update_qh(op::OrderParams, ep::ExtParams) = update_qh(op.ph, op.pdh, op.q, op.p, op.pd, ep.α, ep.αD, ep.β, ep.σ)

function update_ph(p, q, pd, β, σ)
    κ1, κ2, κs = compute_κ(σ)
    kss = κs^2
    k1s = κ1^2

    (β * k1s * (k1s*p + kss*q)) / (1 + β * k1s * (p - pd) + β * kss * (q-1))^2
end
update_ph(op::OrderParams, ep::ExtParams) = update_ph(op.p,op.q,op.pd,ep.β,ep.σ)

function update_pdh(p, q, pd, β, σ)
    κ1, κ2, κs = compute_κ(σ)
    kss = κs^2
    k1s = κ1^2

    (k1s*(1 + β * k1s*(2*p - pd) + β * kss * (2*q-1))) / (1 + β * k1s * (p - pd) + β * kss * (q-1))^2
end
update_pdh(op::OrderParams, ep::ExtParams) = update_pdh(op.p,op.q,op.pd,ep.β,ep.σ)

function update_p(m,q,ph,pdh,α,αD,β)
    αT = α/αD

    (m^2/αD) + (q + αT * β * ph*(1-q)^2)/(αT * β * (q-1) * (ph - pdh) - 1)^2
end
update_p(op::OrderParams, ep::ExtParams) = update_p(op.m,op.q,op.ph,op.pdh,ep.α,ep.αD,ep.β)

function update_pd(ph,pdh,q,m,α,αD,β)
    αT = α/αD

    (m^2/αD) + (1 + αT * β * (2*ph - pdh)*(1-q)^2) / (αT * β * (q-1) * (ph - pdh) - 1)^2
end
update_pd(op::OrderParams, ep::ExtParams) = update_pd(op.ph,op.pdh,op.q,op.m,ep.α,ep.αD,ep.β)

function update_mh(ph,pdh,m,α,αD)
    αT = α/αD
    αT *(pdh - ph) * m
end
update_mh(op::OrderParams, ep::ExtParams) = update_mh(op.ph,op.pdh,op.m,ep.α,ep.αD)

#### ENERGETIC TERM ################################################
function Ge(qh, mh, α, β, σ)
    
    -∫D(z -> begin
        term = √(α * qh) * z + mh
        log(2 * cosh(β * term))
    end) / β
end
Ge(op::OrderParams, ep::ExtParams) = Ge(op.qh, op.mh, ep.α, ep.β)

#### CORRESPONDING UPDATES

function update_q(qh, mh, α, β)
    ∫D(z -> begin
        term = √(α * qh) * z + mh 
        tanh(β * term)^2
    end)
end
update_q(op::OrderParams, ep::ExtParams) = update_q(op.qh, op.mh, ep.α, ep.β)

function update_m(qh, mh, α, β)
    ∫D(z -> begin
        term = √(α * qh) * z + mh
        tanh(β * term)
    end)
end
update_m(op::OrderParams, ep::ExtParams) = update_m(op.qh, op.mh, ep.α, ep.β)


############ Thermodynamic functions ############

function free_energy(op::OrderParams, ep::ExtParams)
    #Gi(op, ep) + Gs1(op, ep) + Gs2(op, ep) + Ge(op, ep)
    return -99
end

function energy(q, qh, p, ph, pd, pdh, m, mh, α, αD, β, σ)
    κ1, κ2, κs = compute_κ(σ)
    kss = κs^2
    k1s = κ1^2
    
    Σ = kss*q + k1s*p
    Σd = kss + k1s*pd 

    αT = α/αD

    0.5*(α*(1 - p*ph + pd*pdh + 2*(1 - q)*qh*β) + m*((ph - pdh)*αT*m + 2*mh)) + 0.5*α*(β*(Σ-Σd)^2 - Σd)/(1+β*(Σ - Σd))^2 +
    + 0.5*α*(ph*q - pdh + αT*β*(1-q)^2 * (pdh -ph)^2)/(1-αT*β*(pdh-ph)*(1-q))^2 - mh*∫D(z -> tanh(β*(z*√(α*qh) + mh))) -α*qh +
    + α*qh*∫D(z -> tanh(β*(z*√(α*qh) + mh))^2)
end

energy(op::OrderParams, ep::ExtParams) = energy(op.q, op.qh, op.p, op.ph, op.pd, op.pdh, op.m, op.mh, ep.α, ep.αD, ep.β, ep.σ)

function entropy(q, qh, p, ph, pd, pdh, m, mh, α, αD, β, σ)
    κ1, κ2, κs = compute_κ(σ)
    kss = κs^2
    k1s = κ1^2
    
    Σ = kss*q + k1s*p
    Σd = kss + k1s*pd 

    αT = α/αD

    0.5*β*α*(2-2*p*ph+2*pd*pdh+3*(1-q)*β*qh) + β*m*((ph-pdh)*αT*m + 2*mh) +
    + 0.5*α*(log(1 + β*Σ -β*Σd) - β*(Σ + Σd + β*Σd*(Σ - Σd))/(1+β*(Σ - Σd))^2) +
    + 0.5*α*(log(1-αT*β*(pdh-ph)*(1-q))/αT + β*(ph*(3*q-1)-pdh*(1+q)+(1-q)*(pdh-ph)*(pdh-ph*q)*αT*β)/(1-αT*β*(1-q)*(pdh - ph))^2)+
    + ∫D(z -> log(2*cosh(β*(z*√(α*qh) + mh)))) - β*(mh * ∫D(z -> tanh(β*(√(α*qh)*z + mh)))+ α*qh*(1-∫D(z -> tanh(β*(√(α*qh)*z + mh))^2)))
end

entropy(op::OrderParams, ep::ExtParams) = entropy(op.q, op.qh, op.p, op.ph, op.pd, op.pdh, op.m, op.mh, ep.α, ep.αD, ep.β, ep.σ)

## Thermodynamic functions
function all_therm_func(op::OrderParams, ep::ExtParams)
    e = energy(op.q, op.qh, op.p, op.ph, op.pd, op.pdh, op.m, op.mh, ep.α, ep.αD, ep.β, ep.σ)
    s = entropy(op.q, op.qh, op.p, op.ph, op.pd, op.pdh, op.m, op.mh, ep.α, ep.αD, ep.β, ep.σ)
    return ThermFunc(e, s)
end

#################  SADDLE POINT  ##################
# Right-hand-side

fq(op, ep) = update_q(op, ep)			   	   # q = fq (der: qh)
fqh(op, ep) = update_qh(op, ep)		   		   # qh = fqh (der: q)
fp(op, ep) = update_p(op, ep)			   	   # q = fq (der: qh)
fph(op, ep) = update_ph(op, ep)		   		   # qh = fqh (der: q)
fpd(op, ep) = update_pd(op, ep)			   	   # q = fq (der: qh)
fpdh(op, ep) = update_pdh(op, ep)		   	   # qh = fqh (der: q)
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
       	
        @update  op.q    fq       Δ ψ verb  op ep   #update q
        @update  op.qh   fqh      Δ ψ verb  op ep  #update qh
        @update  op.p    fp       Δ ψ verb  op ep   #update q
        @update  op.ph   fph      Δ ψ verb  op ep  #update qh
        @update  op.pd   fpd      Δ ψ verb  op ep   #update q
        @update  op.pdh  fpdh     Δ ψ verb  op ep  #update qh
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
    q = 0.5, qh = 0.2, 
    p = 0.5, ph = 0.2, 
    pd = 0.5, pdh = 0.2, 
    m = 0.3, mh = 0.3,
    α = 0.1, αD = 0.1, β = 1.0, σ = :sign, 
    ϵ = 1e-5, maxiters = 10000, verb = 3, ψ = 0.,
    kws...)

    op = OrderParams(q,qh,p,ph,pd,pdh,m,mh)
    ep = ExtParams(α,αD,β,σ)
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
    p = 0.5, ph = 0.2, 
    pd = 0.5, pdh = 0.2, 
    m = 0.3, mh = 0.3,
    α = 0.1, αD = 0.1, β = 1.0,  σ = :sign,
    k1s = 0.101321, kss= 0.297621,
    ϵ = 1e-5, maxiters = 10000, verb = 3, ψ = 0.,
    kws...)

    op = OrderParams(q,qh,p,ph,pd,pdh,m,mh)
    ep = ExtParams(first(α),first(αD),first(β),σ)
    pars = Params(ϵ, ψ, maxiters, verb)

    return span!(op, ep, pars; α=α, αD=αD, β=β, kws...)
end

function span!(op::OrderParams, ep::ExtParams, pars::Params;
        α=0.2, αD=0.1, β=1.0, 
        resfile = "hopfield_RF_aD.txt")

    !isfile(resfile) && open(resfile, "w") do f
        allheadersshow(f, ExtParams, OrderParams, ThermFunc)
    end

    results = []

    for β in β, α in α, αD in αD
        ep.α = α;
        ep.αD = αD;
        ep.β = β;

        println("# NEW ITER: α=$(ep.α) αD=$(ep.αD) β=$(ep.β)")
        
	    ok, Δ = converge!(op, ep, pars)

        println("converge! done")

        tf = all_therm_func(op, ep)

        push!(results, (ok, deepcopy(ep), deepcopy(op), deepcopy(tf)))

        ok && open(resfile, "a") do rf
            println(rf, plainshow(ep), " ", plainshow(op), " ", plainshow(tf))
        end

        !ok && break
        pars.verb > 0 && print(ep, "\n", tf,"\n")
    end
    return results
end

#q, qh, p, ph, pd, pdh, m, mh, α, αD, β, σ
end #module
