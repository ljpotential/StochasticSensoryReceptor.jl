
const dt = 0.01 # time step
const epsilon = 0.01 # for WCA potential
const sigma = 1.5 # for WCA potential

""" binary sequence s(t), and the position of particles """
mutable struct Results
    output::Vector{Int}
    pos_x::Vector{Float64}
    pos_y::Vector{Float64}
    pos_z::Vector{Float64}
    pos_rec::Vector{Float64}
end

""" frequency data for run_brownian_freq """
mutable struct Results_freq
    freq::Vector{Int}
end

""" frequency data for run_brownian_freq_all """
mutable struct Results_freq_all
    freq::Dict{Int,Vector{Int}}
end

"""
    run_brownian(Nl, box_x, box_y, box_z, gamma, alpha, Temp, vx, cutoff, Eon, Eoff, Eb, Nt)

Compute the binding-unbinding events modeled as free Brownian particles.

It will return binary sequence s(t) and the particle positions in the last step.

# Arguments
- `Nl::Int`: the number of ligand
- `box_x::Float64`, `box_y::Float64`, `box_z::Float64`: dimension of box
- `gamma::Float64`: friction coefficient
- `alpha::Float64`: ligand displacement from the receptor in unbinding event
- `Temp::Float64`: temperature
- `vx::Float64`: receptor velocity
- `cutoff::Float64`: cut-off radius for receptor
- `Eon::Float64`: energy of the binding state
- `Eoff::Float64`: energy of the unbinding state
- `Eb::Float64`: energy of the barrier
- `Nt::Int`: total time steps
"""
function run_brownian(Nl::Int, box_x::Float64, box_y::Float64, box_z::Float64,
    gamma::Float64, alpha::Float64, Temp::Float64, vx::Float64, cutoff::Float64,
    Eon::Float64, Eoff::Float64, Eb::Float64, Nt::Int)
    # building ligands and a single receptor in the box
    output = zeros(Int, Nt) # counts
    D_box = Float64[box_x, box_y, box_z] # box dimension
    pos_x = D_box[1] .* rand(Nl) # X positions of ligands
    pos_y = D_box[2] .* rand(Nl) # Y positions of ligands 
    pos_z = D_box[3] .* rand(Nl) # Z positions of ligands 
    pos_rec = D_box / 2 # initial position of receptor (center)
    noise_x = Vector{Float64}(undef, Nl) # pre-allocation of noise X
    noise_y = Vector{Float64}(undef, Nl) # pre-allocation of noise Y
    noise_z = Vector{Float64}(undef, Nl) # pre-allocation of noise Z
    idx_lig = [0] # 0 for unbound state, or index of a ligand (bound)
    # equilibrium 
    println("1. Equilibrium step")
    eq_step = 1000000 # with 1 million steps
    p = Progress(eq_step, 1) # for progress bar
    for i = 1:eq_step
        update_idx_output!(idx_lig, pos_x, pos_y, pos_z, pos_rec, D_box, gamma, alpha, Temp,
            vx, cutoff, Eon, Eoff, Eb)
        update_pos_lig!(pos_x, pos_y, pos_z, D_box, noise_x, noise_y, noise_z, gamma, Temp)
        update_pos_rec!(pos_rec, pos_x, pos_y, pos_z, D_box, vx, idx_lig)
        next!(p) # for progress bar
    end
    # production run
    println("2. Production run")
    p = Progress(Nt, 1) # for progress bar
    for i = 1:Nt
        update_idx_output!(output, idx_lig, pos_x, pos_y, pos_z, pos_rec, D_box, gamma,
            alpha, Temp, vx, cutoff, Eon, Eoff, Eb, i)
        update_pos_lig!(pos_x, pos_y, pos_z, D_box, noise_x, noise_y, noise_z, gamma, Temp)
        update_pos_rec!(pos_rec, pos_x, pos_y, pos_z, D_box, vx, idx_lig)
        next!(p) # for progress bar
    end
    return Results(output, pos_x, pos_y, pos_z, pos_rec)
end

"""
    run_brownian_traj(Nl, box_x, box_y, box_z, gamma, alpha, Temp, vx, cutoff, Eon, Eoff,
                      Eb, Nt, freq, file_name)

Compute the binding-unbinding events modeled as free Brownian particles

It will return binary sequence s(t) and the particle positions in the last step.
It will also produce a XYZ-type trajectory file for ligand & receptor particles.

# Arguments
- `Nl::Int`: the number of ligand
- `box_x::Float64`, `box_y::Float64`, `box_z::Float64`: dimension of box
- `gamma::Float64`: friction coefficient
- `alpha::Float64`: ligand displacement from the receptor in unbinding event
- `Temp::Float64`: temperature
- `vx::Float64`: receptor velocity
- `cutoff::Float64`: cut-off radius for receptor
- `Eon::Float64`: energy of the binding state
- `Eoff::Float64`: energy of the unbinding state
- `Eb::Float64`: energy of the barrier
- `Nt::Int`: total time steps
- `freq::Int`: save trajectory every "freq" time step
- `file_name::String`: file name for trajectory
"""
function run_brownian_traj(Nl::Int, box_x::Float64, box_y::Float64, box_z::Float64,
    gamma::Float64, alpha::Float64, Temp::Float64, vx::Float64, cutoff::Float64,
    Eon::Float64, Eoff::Float64, Eb::Float64, Nt::Int, freq::Int, file_name::String)
    # building ligands and a single receptor in the box
    output = zeros(Int, Nt) # counts
    D_box = Float64[box_x, box_y, box_z] # box dimension
    pos_x = D_box[1] .* rand(Nl) # X positions of ligands
    pos_y = D_box[2] .* rand(Nl) # Y positions of ligands 
    pos_z = D_box[3] .* rand(Nl) # Z positions of ligands 
    pos_rec = D_box / 2 # initial position of receptor (center)
    noise_x = Vector{Float64}(undef, Nl) # pre-allocation of noise X
    noise_y = Vector{Float64}(undef, Nl) # pre-allocation of noise Y
    noise_z = Vector{Float64}(undef, Nl) # pre-allocation of noise Z
    idx_lig = [0] # 0 for unbound state, or index of a ligand (bound)
    # equilibrium 
    println("1. Equilibrium step")
    eq_step = 1000000 # with 1 million steps
    p = Progress(eq_step, 1) # for progress bar
    for i = 1:eq_step
        update_idx_output!(idx_lig, pos_x, pos_y, pos_z, pos_rec, D_box, gamma, alpha, Temp,
            vx, cutoff, Eon, Eoff, Eb)
        update_pos_lig!(pos_x, pos_y, pos_z, D_box, noise_x, noise_y, noise_z, gamma, Temp)
        update_pos_rec!(pos_rec, pos_x, pos_y, pos_z, D_box, vx, idx_lig)
        next!(p) # for progress bar
    end
    # production run
    println("2. Production run")
    io = open(string(file_name, ".xyz"), "w") # trajectory file
    p = Progress(Nt, 1) # for progress bar
    for i = 1:Nt
        update_idx_output!(output, idx_lig, pos_x, pos_y, pos_z, pos_rec, D_box, gamma,
            alpha, Temp, vx, cutoff, Eon, Eoff, Eb, i)
        update_pos_lig!(pos_x, pos_y, pos_z, D_box, noise_x, noise_y, noise_z, gamma, Temp)
        update_pos_rec!(pos_rec, pos_x, pos_y, pos_z, D_box, vx, idx_lig)
        if freq == 0 # no trajectory
        elseif rem(i, freq) == 0 # ever freq step
            write_trj!(pos_x, pos_y, pos_z, pos_rec, io)
        end
        next!(p) # for progress bar
    end
    close(io)
    if freq == 0 # remove the blank trajectory
        rm(string(file_name, ".xyz"))
    end
    return Results(output, pos_x, pos_y, pos_z, pos_rec)
end

"""
    run_brownian_freq(Nl, box_x, box_y, box_z, gamma, alpha, Temp, vx, cutoff, Eon, Eoff,
                      Eb, Nt, n_t, time_lag)

Compute the binding-unbinding events modeled as free Brownian particles

It will return frequency data for n-point trajectories (n\\_t) with time_lag.

# Examples
when n\\_t = 2, freq = [freq00, freq01, freq10, freq11]

# Arguments
- `Nl::Int`: the number of ligand
- `box_x::Float64`, `box_y::Float64`, `box_z::Float64`: dimension of box
- `gamma::Float64`: friction coefficient
- `alpha::Float64`: ligand displacement from the receptor in unbinding event
- `Temp::Float64`: temperature
- `vx::Float64`: receptor velocity
- `cutoff::Float64`: cut-off radius for receptor
- `Eon::Float64`: energy of the binding state
- `Eoff::Float64`: energy of the unbinding state
- `Eb::Float64`: energy of the barrier
- `Nt::Int`: total time steps
- `n_t::Int`: n-point trajectories
- `time_lag::Int`: time lag
"""
function run_brownian_freq(Nl::Int, box_x::Float64, box_y::Float64, box_z::Float64,
    gamma::Float64, alpha::Float64, Temp::Float64, vx::Float64, cutoff::Float64,
    Eon::Float64, Eoff::Float64, Eb::Float64, Nt::Int, n_t::Int, time_lag::Int)
    # building ligands and a single receptor in the box
    D_box = Float64[box_x, box_y, box_z] # box dimension
    pos_x = D_box[1] .* rand(Nl) # X positions of ligands
    pos_y = D_box[2] .* rand(Nl) # Y positions of ligands 
    pos_z = D_box[3] .* rand(Nl) # Z positions of ligands 
    pos_rec = D_box / 2 # initial position of receptor (center)
    noise_x = Vector{Float64}(undef, Nl) # pre-allocation of noise X
    noise_y = Vector{Float64}(undef, Nl) # pre-allocation of noise Y
    noise_z = Vector{Float64}(undef, Nl) # pre-allocation of noise Z
    idx_lig = [0] # 0 for unbound state, or index of a ligand (bound)
    seg = zeros(Int, time_lag, n_t) # pre-allocation of seg
    freq = zeros(Int, 2^n_t) # output
    # equilibrium 
    println("1. Equilibrium step")
    eq_step = 1000000 # with 1 million steps
    p = Progress(eq_step, 1) # for progress bar
    for i = 1:eq_step
        update_idx_output!(idx_lig, pos_x, pos_y, pos_z, pos_rec, D_box, gamma, alpha, Temp,
            vx, cutoff, Eon, Eoff, Eb)
        update_pos_lig!(pos_x, pos_y, pos_z, D_box, noise_x, noise_y, noise_z, gamma, Temp)
        update_pos_rec!(pos_rec, pos_x, pos_y, pos_z, D_box, vx, idx_lig)
        next!(p) # for progress bar
    end
    # production run
    println("2. Production run")
    p = Progress(Nt, 1) # for progress bar
    for i = 1:Nt
        output = update_idx!(idx_lig, pos_x, pos_y, pos_z, pos_rec, D_box, gamma, alpha, vx,
            Temp, cutoff, Eon, Eoff, Eb)
        count_freq!(seg, freq, i, output, n_t, time_lag)
        update_pos_lig!(pos_x, pos_y, pos_z, D_box, noise_x, noise_y, noise_z, gamma, Temp)
        update_pos_rec!(pos_rec, pos_x, pos_y, pos_z, D_box, vx, idx_lig)
        next!(p) # for progress bar
    end
    return Results_freq(freq)
end

"""
    run_brownian_freq_all(Nl, box_x, box_y, box_z, gamma, alpha, Temp, vx, cutoff, Eon, 
                          Eoff, Eb, Nt, n_t_vec, time_lag)

Compute the binding-unbinding events modeled as free Brownian particles

It will return all frequency data for n-point traj (n\\_t\\_vec) with time_lag.

# Examples
when n\\_t\\_vec = [2, 3],
freq = Dict[2 => [freq00, freq01, freq10, freq11],
            3 => [freq000, freq001, freq010, freq011, freq100, freq101,
                  freq110, freq111]]

# Arguments
- `Nl::Int`: the number of ligand
- `box_x::Float64`, `box_y::Float64`, `box_z::Float64`: dimension of box
- `gamma::Float64`: friction coefficient
- `alpha::Float64`: ligand displacement from the receptor in unbinding event
- `Temp::Float64`: temperature
- `vx::Float64`: receptor velocity
- `cutoff::Float64`: cut-off radius for receptor
- `Eon::Float64`: energy of the binding state
- `Eoff::Float64`: energy of the unbinding state
- `Eb::Float64`: energy of the barrier
- `Nt::Int`: total time steps
- `n_t_vec::Vector{Int}`: n-point trajectories
- `time_lag::Int`: time lag
"""
function run_brownian_freq_all(Nl::Int, box_x::Float64, box_y::Float64, box_z::Float64,
    gamma::Float64, alpha::Float64, Temp::Float64, vx::Float64, cutoff::Float64,
    Eon::Float64, Eoff::Float64, Eb::Float64, Nt::Int, n_t_vec::Vector{Int}, time_lag::Int)
    # building ligands and a single receptor in the box
    D_box = Float64[box_x, box_y, box_z] # box dimension
    pos_x = D_box[1] .* rand(Nl) # X positions of ligands
    pos_y = D_box[2] .* rand(Nl) # Y positions of ligands 
    pos_z = D_box[3] .* rand(Nl) # Z positions of ligands 
    pos_rec = D_box / 2 # initial position of receptor (center)
    noise_x = Vector{Float64}(undef, Nl) # pre-allocation of noise X
    noise_y = Vector{Float64}(undef, Nl) # pre-allocation of noise Y
    noise_z = Vector{Float64}(undef, Nl) # pre-allocation of noise Z
    idx_lig = [0] # 0 for unbound state, or index of a ligand (bound)
    seg = Dict(n_t => zeros(Int, time_lag, n_t) for n_t in n_t_vec)
    freq = Dict(n_t => zeros(Int, 2^n_t) for n_t in n_t_vec) # output
    # equilibrium 
    println("1. Equilibrium step")
    eq_step = 1000000 # with 1 million steps
    p = Progress(eq_step, 1) # for progress bar
    for i = 1:eq_step
        update_idx_output!(idx_lig, pos_x, pos_y, pos_z, pos_rec, D_box, gamma, alpha, Temp,
            vx, cutoff, Eon, Eoff, Eb)
        update_pos_lig!(pos_x, pos_y, pos_z, D_box, noise_x, noise_y, noise_z, gamma, Temp)
        update_pos_rec!(pos_rec, pos_x, pos_y, pos_z, D_box, vx, idx_lig)
        next!(p) # for progress bar
    end
    # production run
    println("2. Production run")
    p = Progress(Nt, 1) # for progress bar
    for i = 1:Nt
        output = update_idx!(idx_lig, pos_x, pos_y, pos_z, pos_rec, D_box, gamma, alpha, vx,
            Temp, cutoff, Eon, Eoff, Eb)
        count_freq!(seg, freq, i, output, time_lag)
        update_pos_lig!(pos_x, pos_y, pos_z, D_box, noise_x, noise_y, noise_z, gamma, Temp)
        update_pos_rec!(pos_rec, pos_x, pos_y, pos_z, D_box, vx, idx_lig)
        next!(p) # for progress bar
    end
    return Results_freq_all(freq)
end

"""
    run_wca(Nl, box_x, box_y, box_z, gamma, alpha, Temp, vx, cutoff, Eon, Eoff, Eb, Nt)

Compute the binding-unbinding events modeled with the WCA potential

It will return binary sequence s(t) and the particle positions in the last step

# Arguments
- `Nl::Int`: the number of ligand
- `box_x::Float64`, `box_y::Float64`, `box_z::Float64`: dimension of box
- `gamma::Float64`: friction coefficient
- `alpha::Float64`: ligand displacement from the receptor in unbinding event
- `Temp::Float64`: temperature
- `vx::Float64`: receptor velocity
- `cutoff::Float64`: cut-off radius for receptor
- `Eon::Float64`: energy of the binding state
- `Eoff::Float64`: energy of the unbinding state
- `Eb::Float64`: energy of the barrier
- `Nt::Int`: total time steps
"""
function run_wca(Nl::Int, box_x::Float64, box_y::Float64, box_z::Float64, gamma::Float64,
    alpha::Float64, Temp::Float64, vx::Float64, cutoff::Float64, Eon::Float64,
    Eoff::Float64, Eb::Float64, Nt::Int)
    # building ligands and a single receptor in the box
    output = zeros(Int, Nt) # counts
    D_box = Float64[box_x, box_y, box_z] # box dimension
    pos_x = D_box[1] .* rand(Nl) # X positions of ligands
    pos_y = D_box[2] .* rand(Nl) # Y positions of ligands 
    pos_z = D_box[3] .* rand(Nl) # Z positions of ligands 
    pos_rec = D_box / 2 # initial position of receptor (center)
    force_x = Vector{Float64}(undef, Nl) # pre-allocation of force
    force_y = Vector{Float64}(undef, Nl) # pre-allocation of force
    force_z = Vector{Float64}(undef, Nl) # pre-allocation of force
    noise_x = Vector{Float64}(undef, Nl) # pre-allocation of noise X
    noise_y = Vector{Float64}(undef, Nl) # pre-allocation of noise Y
    noise_z = Vector{Float64}(undef, Nl) # pre-allocation of noise Z
    kloop, jloop = kj_loop(Nl) # for update_force!
    idx_lig = [0] # 0 for unbound state, or index of a ligand (bound)
    # Energy minimization
    println("0. Energy Minimization")
    steepest_descent!(pos_x, pos_y, pos_z, force_x, force_y, force_z, D_box, kloop, jloop,
        1000)
    # equilibrium with 1 million steps
    println("1. Equilibrium step")
    eq_step = 1000000 # with 1 million steps
    p = Progress(eq_step, 1) # for progress bar
    for i = 1:eq_step
        update_idx_output_wca!(idx_lig, pos_x, pos_y, pos_z, pos_rec, D_box, gamma, alpha,
            Temp, vx, cutoff, Eon, Eoff, Eb) # no output
        update_force!(force_x, force_y, force_z, pos_x, pos_y, pos_z, D_box, kloop, jloop)
        update_pos_lig!(pos_x, pos_y, pos_z, force_x, force_y, force_z, D_box, noise_x,
            noise_y, noise_z, gamma, Temp)
        update_pos_rec!(pos_rec, pos_x, pos_y, pos_z, D_box, vx, idx_lig)
        next!(p) # for progress bar
    end
    # production run
    println("2. Production run")
    p = Progress(Nt, 1) # for progress bar
    for i = 1:Nt # iteration
        update_idx_output_wca!(output, idx_lig, pos_x, pos_y, pos_z, pos_rec, D_box, gamma,
            alpha, Temp, vx, cutoff, Eon, Eoff, Eb, i)
        update_force!(force_x, force_y, force_z, pos_x, pos_y, pos_z, D_box, kloop, jloop)
        update_pos_lig!(pos_x, pos_y, pos_z, force_x, force_y, force_z, D_box, noise_x,
            noise_y, noise_z, gamma, Temp)
        update_pos_rec!(pos_rec, pos_x, pos_y, pos_z, D_box, vx, idx_lig)
        next!(p) # for progress bar
    end
    return Results(output, pos_x, pos_y, pos_z, pos_rec)
end

"""
    run_wca_traj(Nl, box_x, box_y, box_z, gamma, alpha, Temp, vx, cutoff, Eon, Eoff, Eb, Nt,
                 freq, file_name)

Compute the binding-unbinding events modeled with the WCA potential

It will return binary sequence s(t) and the particle positions in the last step
It will also produce a XYZ-type trajectory file for ligand & receptor particles

# Arguments
- `Nl::Int`: the number of ligand
- `box_x::Float64`, `box_y::Float64`, `box_z::Float64`: dimension of box
- `gamma::Float64`: friction coefficient
- `alpha::Float64`: ligand displacement from the receptor in unbinding event
- `Temp::Float64`: temperature
- `vx::Float64`: receptor velocity
- `cutoff::Float64`: cut-off radius for receptor
- `Eon::Float64`: energy of the binding state
- `Eoff::Float64`: energy of the unbinding state
- `Eb::Float64`: energy of the barrier
- `Nt::Int`: total time steps
- `freq::Int`: save trajectory every "freq" time step
- `file_name::String`: file name for trajectory
"""
function run_wca_traj(Nl::Int, box_x::Float64, box_y::Float64, box_z::Float64,
    gamma::Float64, alpha::Float64, Temp::Float64, vx::Float64, cutoff::Float64,
    Eon::Float64, Eoff::Float64, Eb::Float64, Nt::Int, freq::Int, file_name::String)
    # building ligands and a single receptor in the box
    output = zeros(Int, Nt) # counts
    D_box = Float64[box_x, box_y, box_z] # box dimension
    pos_x = D_box[1] .* rand(Nl) # X positions of ligands
    pos_y = D_box[2] .* rand(Nl) # Y positions of ligands 
    pos_z = D_box[3] .* rand(Nl) # Z positions of ligands 
    pos_rec = D_box / 2 # initial position of receptor (center)
    force_x = Vector{Float64}(undef, Nl) # pre-allocation of force
    force_y = Vector{Float64}(undef, Nl) # pre-allocation of force
    force_z = Vector{Float64}(undef, Nl) # pre-allocation of force
    noise_x = Vector{Float64}(undef, Nl) # pre-allocation of noise X
    noise_y = Vector{Float64}(undef, Nl) # pre-allocation of noise Y
    noise_z = Vector{Float64}(undef, Nl) # pre-allocation of noise Z
    kloop, jloop = kj_loop(Nl) # for update_force!
    idx_lig = [0] # 0 for unbound state, or index of a ligand (bound)
    # Energy minimization
    println("0. Energy Minimization")
    steepest_descent!(pos_x, pos_y, pos_z, force_x, force_y, force_z, D_box, kloop, jloop,
        1000)
    # equilibrium with 1 million steps
    println("1. Equilibrium step")
    eq_step = 1000000 # with 1 million steps
    p = Progress(eq_step, 1) # for progress bar
    for i = 1:eq_step
        update_idx_output_wca!(idx_lig, pos_x, pos_y, pos_z, pos_rec, D_box, gamma, alpha,
            Temp, vx, cutoff, Eon, Eoff, Eb) # no output
        update_force!(force_x, force_y, force_z, pos_x, pos_y, pos_z, D_box, kloop, jloop)
        update_pos_lig!(pos_x, pos_y, pos_z, force_x, force_y, force_z, D_box, noise_x,
            noise_y, noise_z, gamma, Temp)
        update_pos_rec!(pos_rec, pos_x, pos_y, pos_z, D_box, vx, idx_lig)
        next!(p) # for progress bar
    end
    # production run
    println("2. Production run")
    io = open(string(file_name, ".xyz"), "w") # trajectory file
    p = Progress(Nt, 1) # for progress bar
    for i = 1:Nt # iteration
        update_idx_output_wca!(output, idx_lig, pos_x, pos_y, pos_z, pos_rec, D_box, gamma,
            alpha, Temp, vx, cutoff, Eon, Eoff, Eb, i)
        update_force!(force_x, force_y, force_z, pos_x, pos_y, pos_z, D_box, kloop, jloop)
        update_pos_lig!(pos_x, pos_y, pos_z, force_x, force_y, force_z, D_box, noise_x,
            noise_y, noise_z, gamma, Temp)
        update_pos_rec!(pos_rec, pos_x, pos_y, pos_z, D_box, vx, idx_lig)
        if freq == 0 # no trajectory
        elseif rem(i, freq) == 0 # ever freq step
            write_trj!(pos_x, pos_y, pos_z, pos_rec, io)
        end
        next!(p) # for progress bar
    end
    close(io)
    if freq == 0 # remove the blank trajectory
        rm(string(file_name, ".xyz"))
    end
    return Results(output, pos_x, pos_y, pos_z, pos_rec)
end