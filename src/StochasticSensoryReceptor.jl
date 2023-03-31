# Julia code for simulation and analysis of stochastic sensory receptor
# Version 1.0

# Please refer to this paper:
# A. Pagare, S. H. Min, and Z. Lu,
# "Theoretical upper bound of multiplexing in stochastic sensory receptors",
# Phys. Rev. Research, X, XXXXX (2023).

module StochasticSensoryReceptor

using ProgressMeter
using DelimitedFiles
using Random
using StatsBase
using LinearAlgebra

export run_brownian
export run_brownian_traj
export run_brownian_freq
export run_brownian_freq_all
export run_wca
export run_wca_traj
export trj2seg, seg2decimal, decimal2freq, calc_llf

include("run_functions.jl")
include("sub_functions.jl")
include("mle_analysis.jl")

end # module