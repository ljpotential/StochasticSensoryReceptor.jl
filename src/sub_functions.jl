# for free Brownian motion

""" update idx_lig, position of the bound ligand, and return output::Int """
function update_idx!(idx_lig::Vector{Int}, pos_x::Vector{Float64},
    pos_y::Vector{Float64}, pos_z::Vector{Float64}, pos_rec::Vector{Float64},
    D_box::Vector{Float64}, gamma::Float64, alpha::Float64, vx::Float64,
    Temp::Float64, cutoff::Float64, Eon::Float64, Eoff::Float64, Eb::Float64)
    within_lig = Int.([])
    output = 0
    if idx_lig[1] == 0
        # finding ligands within cutoff radius
        @inbounds for k in eachindex(pos_x) # 1:Nl
            dx = pos_x[k] - pos_rec[1]
            dx -= D_box[1] * round(dx / D_box[1])  # for PBC
            dy = pos_y[k] - pos_rec[2]
            dz = pos_z[k] - pos_rec[3]
            if abs(dx) <= cutoff && abs(dy) <= cutoff && abs(dz) <= cutoff
                dis = sqrt(dx^2 + dy^2 + dz^2)
                if dis <= cutoff
                    push!(within_lig, k)
                end
            end
        end
        if length(within_lig) != 0
            P_on = length(within_lig) * dt * exp(-(Eb - Eoff) / Temp)
            if rand() <= P_on
                idx_lig[1] = rand(within_lig)
                output = 1
                pos_x[idx_lig[1]] = pos_rec[1]
                pos_y[idx_lig[1]] = pos_rec[2]
                pos_z[idx_lig[1]] = pos_rec[3]
            end
        end
    else
        P_off = dt * exp(-(Eb - Eon - (gamma * alpha * vx)) / Temp)
        if rand() <= P_off
            idx_lig[1] = 0
        else
            output = 1
        end
    end
    return output
end

function count_freq!(seg::Array{Int,2}, freq::Vector{Int}, i::Int, output::Int,
    n_t::Int, time_lag::Int)
    idx = get_idx(i, n_t, time_lag)
    seg[idx] = output
    x, y = get_coord(idx, time_lag)
    decimal = get_decimal(seg, x, y, n_t)
    if i > (n_t - 1) * time_lag
        freq[decimal] += 1
    end
    return nothing
end

function count_freq!(seg::Dict{Int,Array{Int,2}}, freq::Dict{Int,Vector{Int}},
    i::Int, output::Int, time_lag::Int)
    for key in keys(seg)
        idx = get_idx(i, key, time_lag)
        seg[key][idx] = output
        x, y = get_coord(idx, time_lag)
        decimal = get_decimal(seg[key], x, y, key)
        if i > (key - 1) * time_lag
            freq[key][decimal] += 1
        end
    end
    return nothing
end

function get_idx(i::Int, n_t::Int, time_lag::Int)
    idx = rem(i, n_t * time_lag)
    if idx == 0
        idx = n_t * time_lag
    end
    return idx
end

function get_coord(idx::Int, time_lag::Int)
    y, x = divrem(idx, time_lag)
    y += 1
    if x == 0
        y -= 1
        x = time_lag
    end
    return x, y
end

function get_decimal(seg::Array{Int,2}, x::Int, y::Int, n_t::Int)
    decimal = 0
    for i = 1:n_t
        y += 1
        if y > n_t
            y = 1
        end
        decimal += seg[x, y] * 2^(n_t - i)
    end
    decimal += 1
    return decimal
end

# for the WCA potential

function kj_loop(Nl::Int)
    kloop = Int[]
    jloop = Int[]
    for k in 1:Nl
        for j in 1:Nl
            if j < k
                push!(kloop, k)
                push!(jloop, j)
            end
        end
    end
    return kloop, jloop
end

""" Energy minimization """
function steepest_descent!(pos_x::Vector{Float64}, pos_y::Vector{Float64},
    pos_z::Vector{Float64}, force_x::Vector{Float64}, force_y::Vector{Float64},
    force_z::Vector{Float64}, D_box::Vector{Float64}, kloop::Vector{Int},
    jloop::Vector{Int}, nstep::Int)
    h = fill(0.1, length(pos_x)) # initial h value, 0.1 is enough for our system
    for i = 1:nstep
        update_force!(force_x, force_y, force_z, pos_x, pos_y, pos_z, D_box,
            kloop, jloop)
        maxF = maximum(sqrt.(force_x .^ 2 + force_y .^ 2 + force_z .^ 2))
        if maxF == 0.0
            println("Energy minimized at $i steps")
            return nothing
        end
        energy_x, energy_y, energy_z = calc_energy(pos_x, pos_y, pos_z, D_box,
            kloop, jloop)
        new_pos_x = pos_x .+ force_x ./ maxF .* h
        new_pos_y = pos_y .+ force_y ./ maxF .* h
        new_pos_z = pos_z .+ force_z ./ maxF .* h
        apply_boundary!(new_pos_x, new_pos_y, new_pos_z, D_box)
        new_energy_x, new_energy_y, new_energy_z = calc_energy(new_pos_x,
            new_pos_y, new_pos_z, D_box, kloop, jloop) # new energy calc
        idx_x = new_energy_x .< energy_x
        idx_y = new_energy_y .< energy_y
        idx_z = new_energy_z .< energy_z
        pos_x[idx_x] = new_pos_x[idx_x] # accept
        pos_y[idx_y] = new_pos_y[idx_y] # accept
        pos_z[idx_z] = new_pos_z[idx_z] # accept
    end
    return nothing
end

""" update force """
function update_force!(force_x::Vector{Float64}, force_y::Vector{Float64},
    force_z::Vector{Float64}, pos_x::Vector{Float64}, pos_y::Vector{Float64},
    pos_z::Vector{Float64}, D_box::Vector{Float64}, kloop::Vector{Int},
    jloop::Vector{Int})
    # initialize
    cutd = 2^(1 / 6) * sigma
    fill!(force_x, 0.0)
    fill!(force_y, 0.0)
    fill!(force_z, 0.0)
    @inbounds for l in eachindex(kloop) # 1:Nl
        dx = pos_x[kloop[l]] - pos_x[jloop[l]]
        dx -= D_box[1] * round(dx / D_box[1])
        dy = pos_y[kloop[l]] - pos_y[jloop[l]]
        dz = pos_z[kloop[l]] - pos_z[jloop[l]]
        # d_lig/2 = 2^(1/6)*sigma
        if abs(dx) <= cutd && abs(dy) <= cutd && abs(dz) <= cutd
            dis = sqrt(dx^2 + dy^2 + dz^2)
            if dis <= cutd
                abs_force = 4 * epsilon * (12 * sigma^12 / dis^13 -
                                           6 * sigma^6 / dis^7)
                fx = abs_force * dx / dis
                fy = abs_force * dy / dis
                fz = abs_force * dz / dis
                force_x[kloop[l]] += fx
                force_y[kloop[l]] += fy
                force_z[kloop[l]] += fz
                force_x[jloop[l]] -= fx
                force_y[jloop[l]] -= fy
                force_z[jloop[l]] -= fz
            end
        end
    end
    return nothing
end

function calc_energy(pos_x::Vector{Float64}, pos_y::Vector{Float64},
    pos_z::Vector{Float64}, D_box::Vector{Float64}, kloop::Vector{Int},
    jloop::Vector{Int})
    # initialize
    cutd = 2^(1 / 6) * sigma
    energy_x = zeros(Float64, length(pos_x))
    energy_y = zeros(Float64, length(pos_x))
    energy_z = zeros(Float64, length(pos_x))
    @inbounds for l in eachindex(kloop)
        dx = pos_x[kloop[l]] - pos_x[jloop[l]]
        dx -= D_box[1] * round(dx / D_box[1])
        dy = pos_y[kloop[l]] - pos_y[jloop[l]]
        dz = pos_z[kloop[l]] - pos_z[jloop[l]]
        # d_lig/2 = 2^(1/6)*sigma
        if abs(dx) <= cutd && abs(dy) <= cutd && abs(dz) <= cutd
            dis = sqrt(dx^2 + dy^2 + dz^2)
            if dis <= cutd
                abs_energy = 4 * epsilon * (sigma^12 / dis^12 -
                                            sigma^6 / dis^6) + epsilon
                ex = abs_energy * dx / dis
                ey = abs_energy * dy / dis
                ez = abs_energy * dz / dis
                energy_x[kloop[l]] += ex
                energy_y[kloop[l]] += ey
                energy_z[kloop[l]] += ez
                energy_x[jloop[l]] -= ex
                energy_y[jloop[l]] -= ey
                energy_z[jloop[l]] -= ez
            end
        end
    end
    return energy_x, energy_y, energy_z
end

""" update idx_ligand without output for equilibrium step (wca)"""
function update_idx_output_wca!(idx_lig::Vector{Int}, pos_x::Vector{Float64},
    pos_y::Vector{Float64}, pos_z::Vector{Float64}, pos_rec::Vector{Float64},
    D_box::Vector{Float64}, gamma::Float64, alpha::Float64, Temp::Float64,
    vx::Float64, cutoff::Float64, Eon::Float64, Eoff::Float64, Eb::Float64)
    within_lig = Int.([])
    if idx_lig[1] == 0
        # finding ligands within cutoff radius
        @inbounds for k in eachindex(pos_x) # 1:Nl
            dx = pos_x[k] - pos_rec[1]
            dx -= D_box[1] * round(dx / D_box[1])  # for PBC
            dy = pos_y[k] - pos_rec[2]
            dz = pos_z[k] - pos_rec[3]
            if abs(dx) <= cutoff && abs(dy) <= cutoff && abs(dz) <= cutoff
                dis = sqrt(dx^2 + dy^2 + dz^2)
                if dis <= cutoff
                    push!(within_lig, k)
                end
            end
        end
        if length(within_lig) == 0
        elseif length(within_lig) == 1
            # output[i] = 0
            P_on = dt * exp(-(Eb - Eoff) / Temp)
            if rand() <= P_on
                idx_lig[1] = within_lig[1]
                # output[i] = 1
                pos_x[idx_lig[1]] = pos_rec[1]
                pos_y[idx_lig[1]] = pos_rec[2]
                pos_z[idx_lig[1]] = pos_rec[3]
                # else
                # output[i] = 0
            end
        else
            # output[i] = 0
            P_on = length(within_lig) * dt * exp(-(Eb - Eoff) / Temp)
            if rand() <= P_on
                idx_lig[1] = calc_closest_lig(pos_x, pos_y, pos_z, pos_rec,
                    within_lig, D_box)
                # output[i] = 1
                pos_x[idx_lig[1]] = pos_rec[1]
                pos_y[idx_lig[1]] = pos_rec[2]
                pos_z[idx_lig[1]] = pos_rec[3]
                # else
                # output[i] = 0
            end
        end
    else
        P_off = dt * exp(-(Eb - Eon - (gamma * alpha * vx)) / Temp)
        if rand() <= P_off
            idx_lig[1] = 0
            # output[i] = 0
            # else
            #    output[i] = 1
        end
    end
    return nothing
end

""" update idx_ligand and output (wca)"""
function update_idx_output_wca!(output::Vector{Int}, idx_lig::Vector{Int},
    pos_x::Vector{Float64}, pos_y::Vector{Float64}, pos_z::Vector{Float64},
    pos_rec::Vector{Float64}, D_box::Vector{Float64}, gamma::Float64,
    alpha::Float64, Temp::Float64, vx::Float64, cutoff::Float64,
    Eon::Float64, Eoff::Float64, Eb::Float64, i::Int)
    within_lig = Int.([])
    if idx_lig[1] == 0
        # finding ligands within cutoff radius
        @inbounds for k in eachindex(pos_x) # 1:Nl
            dx = pos_x[k] - pos_rec[1]
            dx -= D_box[1] * round(dx / D_box[1])  # for PBC
            dy = pos_y[k] - pos_rec[2]
            dz = pos_z[k] - pos_rec[3]
            if abs(dx) <= cutoff && abs(dy) <= cutoff && abs(dz) <= cutoff
                dis = sqrt(dx^2 + dy^2 + dz^2)
                if dis <= cutoff
                    push!(within_lig, k)
                end
            end
        end
        if length(within_lig) == 0
        elseif length(within_lig) == 1
            P_on = dt * exp(-(Eb - Eoff) / Temp)
            if rand() <= P_on
                idx_lig[1] = within_lig[1]
                output[i] = 1
                pos_x[idx_lig[1]] = pos_rec[1]
                pos_y[idx_lig[1]] = pos_rec[2]
                pos_z[idx_lig[1]] = pos_rec[3]
            else
                output[i] = 0
            end
        else
            P_on = length(within_lig) * dt * exp(-(Eb - Eoff) / Temp)
            if rand() <= P_on
                idx_lig[1] = calc_closest_lig(pos_x, pos_y, pos_z, pos_rec,
                    within_lig, D_box)
                output[i] = 1
                pos_x[idx_lig[1]] = pos_rec[1]
                pos_y[idx_lig[1]] = pos_rec[2]
                pos_z[idx_lig[1]] = pos_rec[3]
            else
                output[i] = 0
            end
        end
    else
        P_off = dt * exp(-(Eb - Eon - (gamma * alpha * vx)) / Temp)
        if rand() <= P_off
            idx_lig[1] = 0
            output[i] = 0
        else
            output[i] = 1
        end
    end
    return nothing
end

function calc_closest_lig(pos_x::Vector{Float64}, pos_y::Vector{Float64},
    pos_z::Vector{Float64}, pos_rec::Vector{Float64}, within_lig::Vector{Int},
    D_box::Vector{Float64})
    closest = Int(0)
    temp = Float64(0)
    for i = eachindex(within_lig)
        k = within_lig[i]
        dx = pos_x[k] - pos_rec[1]
        dx -= D_box[1] * round(dx / D_box[1])  # for PBC
        dy = pos_y[k] - pos_rec[2]
        dz = pos_z[k] - pos_rec[3]
        dis = sqrt(dx^2 + dy^2 + dz^2)
        if closest == 0
            closest = k
            temp = dis
        elseif dis < temp
            closest = k
            temp = dis
        end
    end
    return closest
end

# for both cases

""" update ligand positions """
function update_pos_lig!(pos_x::Vector{Float64}, pos_y::Vector{Float64},
    pos_z::Vector{Float64}, force_x::Vector{Float64}, force_y::Vector{Float64},
    force_z::Vector{Float64}, D_box::Vector{Float64}, noise_x::Vector{Float64},
    noise_y::Vector{Float64}, noise_z::Vector{Float64}, gamma::Float64,
    Temp::Float64)
    randn!(noise_x)
    randn!(noise_y)
    randn!(noise_z)
    noise_k = sqrt(2 * Temp * dt / gamma)
    conserv_k = dt / gamma
    # update positions using the Brownian dynamics
    @inbounds for k in eachindex(pos_x)
        pos_x[k] += conserv_k * force_x[k] + noise_k * noise_x[k]
        pos_y[k] += conserv_k * force_y[k] + noise_k * noise_y[k]
        pos_z[k] += conserv_k * force_z[k] + noise_k * noise_z[k]
    end
    # Apply the reflecting boundary and PBC
    apply_boundary!(pos_x, pos_y, pos_z, D_box)
    return nothing
end

""" update idx_lig and position of the bound ligand for equilibrium step """
function update_idx_output!(idx_lig::Vector{Int}, pos_x::Vector{Float64},
    pos_y::Vector{Float64}, pos_z::Vector{Float64}, pos_rec::Vector{Float64},
    D_box::Vector{Float64}, gamma::Float64, alpha::Float64, Temp::Float64,
    vx::Float64, cutoff::Float64, Eon::Float64, Eoff::Float64, Eb::Float64)
    within_lig = Int.([])
    if idx_lig[1] == 0
        # finding ligands within cutoff radius
        @inbounds for k in eachindex(pos_x) # 1:Nl
            dx = pos_x[k] - pos_rec[1]
            dx -= D_box[1] * round(dx / D_box[1])  # for PBC
            dy = pos_y[k] - pos_rec[2]
            dz = pos_z[k] - pos_rec[3]
            if abs(dx) <= cutoff && abs(dy) <= cutoff && abs(dz) <= cutoff
                dis = sqrt(dx^2 + dy^2 + dz^2)
                if dis <= cutoff
                    push!(within_lig, k)
                end
            end
        end
        if length(within_lig) != 0
            P_on = length(within_lig) * dt * exp(-(Eb - Eoff) / Temp)
            if rand() <= P_on
                idx_lig[1] = rand(within_lig)
                # output[i] = 1
                pos_x[idx_lig[1]] = pos_rec[1]
                pos_y[idx_lig[1]] = pos_rec[2]
                pos_z[idx_lig[1]] = pos_rec[3]
            end
        end
    else
        P_off = dt * exp(-(Eb - Eon - (gamma * alpha * vx)) / Temp)
        if rand() <= P_off
            idx_lig[1] = 0
            # output[i] = 0
            # else
            #    output[i] = 1
        end
    end
    return nothing
end

""" update idx_lig, position of the bound ligand, and output """
function update_idx_output!(output::Vector{Int}, idx_lig::Vector{Int},
    pos_x::Vector{Float64}, pos_y::Vector{Float64}, pos_z::Vector{Float64},
    pos_rec::Vector{Float64}, D_box::Vector{Float64}, gamma::Float64,
    alpha::Float64, Temp::Float64, vx::Float64, cutoff::Float64, Eon::Float64,
    Eoff::Float64, Eb::Float64, i::Int)
    within_lig = Int.([])
    if idx_lig[1] == 0
        # finding ligands within cutoff radius
        @inbounds for k in eachindex(pos_x) # 1:Nl
            dx = pos_x[k] - pos_rec[1]
            dx -= D_box[1] * round(dx / D_box[1])  # for PBC
            dy = pos_y[k] - pos_rec[2]
            dz = pos_z[k] - pos_rec[3]
            if abs(dx) <= cutoff && abs(dy) <= cutoff && abs(dz) <= cutoff
                dis = sqrt(dx^2 + dy^2 + dz^2)
                if dis <= cutoff
                    push!(within_lig, k)
                end
            end
        end
        if length(within_lig) == 0
            output[i] = 0
        else
            P_on = length(within_lig) * dt * exp(-(Eb - Eoff) / Temp)
            if rand() <= P_on
                idx_lig[1] = rand(within_lig)
                output[i] = 1
                pos_x[idx_lig[1]] = pos_rec[1]
                pos_y[idx_lig[1]] = pos_rec[2]
                pos_z[idx_lig[1]] = pos_rec[3]
            else
                output[i] = 0
            end
        end
    else
        P_off = dt * exp(-(Eb - Eon - (gamma * alpha * vx)) / Temp)
        if rand() <= P_off
            idx_lig[1] = 0
            output[i] = 0
        else
            output[i] = 1
        end
    end
    return nothing
end

""" update ligand positions """
function update_pos_lig!(pos_x::Vector{Float64}, pos_y::Vector{Float64},
    pos_z::Vector{Float64}, D_box::Vector{Float64}, noise_x::Vector{Float64},
    noise_y::Vector{Float64}, noise_z::Vector{Float64}, gamma::Float64,
    Temp::Float64)
    randn!(noise_x)
    randn!(noise_y)
    randn!(noise_z)
    noise_k = sqrt(2 * Temp * dt / gamma)
    # update positions using the Brownian dynamics
    @inbounds for k in eachindex(pos_x)
        pos_x[k] += noise_k * noise_x[k]
        pos_y[k] += noise_k * noise_y[k]
        pos_z[k] += noise_k * noise_z[k]
    end
    # Apply the reflecting boundary and PBC
    apply_boundary!(pos_x, pos_y, pos_z, D_box)
    return nothing
end

""" apply boundary conditions (x: PBC, y and z: reflecting) """
function apply_boundary!(pos_x::Vector{Float64}, pos_y::Vector{Float64},
    pos_z::Vector{Float64}, D_box::Vector{Float64})
    # Apply the reflecting boundary and PBC
    @inbounds for k in eachindex(pos_x)
        # PBC x
        if pos_x[k] > D_box[1]
            pos_x[k] -= D_box[1]
        elseif pos_x[k] <= 0
            pos_x[k] += D_box[1]
        end
        # reflecting Y
        if pos_y[k] > D_box[2]
            pos_y[k] -= 2 * (pos_y[k] - D_box[2])
        elseif pos_y[k] <= 0
            pos_y[k] = -pos_y[k]
        end
        # reflecting Z
        if pos_z[k] > D_box[3]
            pos_z[k] -= 2 * (pos_z[k] - D_box[3])
        elseif pos_z[k] <= 0
            pos_z[k] = -pos_z[k]
        end
    end
    return nothing
end

""" update positions of receptor and bound ligand (if present) """
function update_pos_rec!(pos_rec::Vector{Float64}, pos_x::Vector{Float64},
    pos_y::Vector{Float64}, pos_z::Vector{Float64}, D_box::Vector{Float64},
    vx::Float64, idx_lig::Vector{Int})
    # only along X direction
    pos_rec[1] += dt * vx
    # PBC x
    if pos_rec[1] > D_box[1]
        pos_rec[1] -= D_box[1]
    elseif pos_rec[1, 1] <= 0
        pos_rec[1] += D_box[1]
    end
    if idx_lig[1] != 0 # update the position of the ligand bound to the receptor
        pos_x[idx_lig[1]] = pos_rec[1]
        pos_y[idx_lig[1]] = pos_rec[2]
        pos_z[idx_lig[1]] = pos_rec[3]
    end
    return nothing
end

""" write trajectory file to XYZ file format """
function write_trj!(pos_x::Vector{Float64}, pos_y::Vector{Float64},
    pos_z::Vector{Float64}, pos_rec::Vector{Float64}, io::IOStream)
    n_particle = length(pos_x) + 1
    write(io, string(n_particle), "\n\n") # number of particle + blank line
    for j in eachindex(pos_x)
        write(io, "C\t") # atom name C for ligand
        writedlm(io, round.([pos_x[j] pos_y[j] pos_z[j]], digits=3))
    end
    write(io, "N\t") # atom name N for receptor
    writedlm(io, round.([pos_rec[1] pos_rec[2] pos_rec[3]], digits=3))
    return nothing
end