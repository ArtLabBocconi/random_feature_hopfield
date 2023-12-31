
## vertical cut
# for αD in 0.030:0.001:0.030

#     P.find_alphac(α=0.500:-0.001:0.300,αD=αD,verb = -1,
#                 σ=:sign,resfile="results/alphac.txt", ψ=0.9, 
#                 dq=3.3410127750224426e-10,qh=105.19065226080156,
#                 dp=3.331297692245768e-10,dph=0.6366197725789939,
#                 pd=100.9999999353968,dpdh=41.16509321703842,
#                 m=0.999999999461402,mh=63.66228837695199,
#                 )

# end

## horizontal cut
for α in 0.700:-0.100:0.200

    P.find_alphac(α=α,αD=0.010:0.001:0.050,verb = -1,
                σ=:sign,resfile="results/alphac.txt", ψ=0.9, 
                dq=3.3410127750224426e-10,qh=105.19065226080156,
                dp=3.331297692245768e-10,dph=0.6366197725789939,
                pd=100.9999999353968,dpdh=41.16509321703842,
                m=0.999999999461402,mh=63.66228837695199,
                )

end