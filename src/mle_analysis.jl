""" Converting trajectory to segment """
function trj2seg(output::Vector{Int}, n_t::Int, time_lag::Int)
    Nt = length(output) # number of total time steps
    ns = Nt - time_lag # number of segments
    seg = zeros(Int, (n_t, ns))
    for i = 1:ns
        seg[1, i] = output[i]
        seg[2, i] = output[i+time_lag]
    end
    return seg
end

""" Converting segment array to decimal vector """
function seg2decimal(seg::Array{Int,2})
    # binary is a column vector of length ls
    binary = [2^(i) for i = 0:(size(seg, 1)-1)]
    decimal = (transpose(binary)*seg)[:] .+ 1
    return decimal
end

""" Collecting frequency for each decimal in the sample space """
function decimal2freq(decimal::Array{Int}, n_t::Int)
    freq = zeros(Int, 2^n_t)
    for i = eachindex(decimal)
        a = decimal[i]
        freq[a] += 1
    end
    return freq
end

""" Calculating log likelihood function """
function calc_llf(freq_star::Vector{Int}, freq::Vector{Int})
    if sum(freq_star) != sum(freq)
        throw(DomainError("samples are not matching in calcLLF"))
    end
    llf = Float64(0)
    tot = sum(freq)
    for i = eachindex(freq_star)
        llf += freq_star[i] * log(freq[i])
    end
    llf = llf / tot - log(tot)
    return llf
end

