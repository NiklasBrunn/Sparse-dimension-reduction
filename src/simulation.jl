#---functions for simulating data:
function addstages!(X::AbstractMatrix{T}, stageno::Int; stagen::Int=1, stagep::Int=2, overlap::Int=1, blockprob::Number=1.0) where T
    curp = 1
    curn = 1
    for i = 1:stageno
        if blockprob < 1.0
            curblock = 1.0*(rand(stagen,stagep) .<= blockprob)
            X[curn:(curn+stagen-1),curp:(curp+stagep-1)] = curblock
        else
            X[curn:(curn+stagen-1),curp:(curp+stagep-1)] .= 1.0
        end
        curp += stagep - overlap
        curn += stagen
    end
    X
end

function sim_scRNAseqData(dataseed = 1;
    n = 1000, 
    num_genes = 50, 
    stageno = 10, 
    stagep = Int(50 / 10), 
    stagen = Int(1000 / 10), 
    stageoverlap = 2, 
    blockprob = 0.6, 
    noiseprob = 0.1, 
    )

    Random.seed!(dataseed)
    X = 1.0*(rand(n, num_genes) .> (1-noiseprob)) 
    X = addstages!(X,stageno,stagen=stagen,stagep=stagep,overlap=stageoverlap,blockprob=blockprob) 
     
    return Float32.(X)
end 