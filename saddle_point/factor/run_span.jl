αD=0.01
resfile="results/span_aD$(αD).txt"

P.span(α=0.300:-0.001:0.050,αD=αD,
                σ=:sign,resfile=resfile, ψ=0.9, 
                dq=3.3410127750224426e-10,qh=105.19065226080156,
                dp=3.331297692245768e-10,dph=0.6366197725789939,
                pd=100.9999999353968,dpdh=41.16509321703842,
                m=0.999999999461402,mh=63.66228837695199,
                )