
for αD in 1.00:-0.10:0.10

    P.find_alphac(α=0.001:0.001:0.150,αD=αD, verb=1,
                σ=:sign,resfile="results/alphac.txt", ψ=0.9, maxiters=1e4,
                dq=8.595353998742104e-9,qh=1.004060267804979,
                dp=1.2894207365173853e-7,dph=0.6366201987745321,
                pd=0.9999999929773609,dphd=0.6366205912296964,
                m=1.0
                )

end

