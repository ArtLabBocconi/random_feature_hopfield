
# for α in 0.1:0.01:0.2  

#     P.find_alphac(α=α,αD=1.0:-0.01:0.1,
#                 σ=:sign,resfile="matteo/alphac_m_sign.txt", ψ=0.5, 
#                 # σ=:id,resfile="matteo/alphac_m_id.txt", ψ=0.5, 
#                 dq=7.450580596925714e-9,qh=1.0040590852585105,
#                 dp=1.0431588916228004e-7,dph=0.6366200967849951,
#                 pd=0.999999993809326,dphd=0.6366203937025133,
#                 m=1.0
#                 )

# end

# 0.113 0.4

# for αD in 0.14:-0.01:0.01
# for αD in 1.00:-0.01:0.01

#     P.find_alphac(α=0.001:0.001:0.15,αD=αD,
#                 σ=:sign,resfile="alphac_2.txt", ψ=0.90, 
#                 # σ=:id,resfile="matteo/alphac_m_id.txt", ψ=0.5, 
#                 dq=7.450580596925714e-9,qh=1.0040590852585105,
#                 dp=1.0431588916228004e-7,dph=0.6366200967849951,
#                 pd=0.999999993809326,dphd=0.6366203937025133,
#                 m=1.0
#                 )

# end


# for αD in 0.133:-0.001:0.001
for αD in 1.000:-0.001:0.001

    P.find_alphac(α=0.0001:0.0001:0.1500,αD=αD,
                # σ=:sign,resfile="matteo/alphac_m_sign.txt", ψ=0.99, 
                σ=:sign,resfile="alphac.txt", ψ=0.5, maxiters=1e6,
                # σ=:id,resfile="matteo/alphac_m_id.txt", ψ=0.5, 
                dq=8.595353998742104e-9,qh=1.004060267804979,
                dp=1.2894207365173853e-7,dph=0.6366201987745321,
                pd=0.9999999929773609,dphd=0.6366205912296964,
                m=1.0
                )

end