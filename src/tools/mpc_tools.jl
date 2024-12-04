using LinearAlgebra


"""
    mpc_to_qp(cost_dict, constraint_dict, system_dict, x0, u_latest, T)

Take a discrete-time finite-horizon optimal control problem and formulate it as a QP.

NOTE: The keys in the return dictionary often overlap with the keys in the argument dictionaries,
but this is by coincidence. The key "Q" means different things within `cost_dict` and the return
dictionary.

# Arguments
- `cost_dict::Dict{String, Array}`: parameters of the cost function. Contains key-value pairs
        - "Q"::Array: the quadratic term of the state penalty (n,n),
        - "q"::Array: the linear term of the state penalty (n,),
        - "R"::Array: the quadratic term of the input penalty (m,m),
        - "r"::Array: the linear term of the input penalty (m,),
        - "S"::Array: the quadratic term of the differenced input penalty (m,),
        - "Q_T"::Array: the quadratic term of the terminal state penalty (n,n),
        - "q_T"::Array: the linear term of the terminal state penalty (n,).
- `constraint_dict::Dict{String, Array}`: parameters of the constraints. Contains key-value pairs
        - "F_x"::Array: the state constraint matrix (c_x,n),
        - "f_x"::Array: the state constraint vector (c_x,),
        - "F_u"::Array: the input constraint matrix (c_u,m),
        - "f_u"::Array: the input constraint vector (c_u,),
        - "F_du"::Array: the differenced input constraint matrix (c_du,m),
        - "f_du"::Array: the differenced input constraint vector (c_du,),
        - "F_T"::Array: the terminal state constraint matrix (c_T,n),
        - "f_T"::Array: the terminal state constraint vector (c_T,).
- `system_dict::Dict{String, Array}`: parameters of the (linearized) system dynamics. Contains key-value pairs
        - "A"::Array: the drift matrix (n,n),
        - "B"::Array: the input matrix (n,m),
        - "w"::Array: the expected disturbance (assumed to be zero) (n,).
- `x0::Array`: the initial state (n,).
- `u_latest::Array`: the latest input applied to the system (m,).
- `T::Integer64`: the prediction horizon.

# Returns 
- `Dict{String, Array}`: parameters of the quadratic program. Contains key-value pairs
        - "Q"::Array: the Hessian of the QP (T*(n+m),T*(n+m)),
        - "q"::Array: the linear term of the QP (T*(n+m),),
        - "G"::Array: the linear inequality constraint matrix (T*(c_x + c_u + c_du),T*(n+m)),
        - "h"::Array: the linear inequality constraint limits (T*(c_x + c_u + c_du),),
        - "A"::Array: the linear equality constraint matrix (T*n,T*(n+m)),
        - "b"::Array: the linear equality constraint limits (T*n,).
"""
function mpc_to_qp(cost_dict, constraint_dict, system_dict, x0, u_latest, T)
    n = size(cost_dict["Q"])[1]
    m = size(cost_dict["R"])[1]

    # Construct QP Hessian
    H = mpc_to_qp_hessian(cost_dict, n, m, T)

    # Construct linear term of QP cost
    g = mpc_to_qp_linear_term(cost_dict, n, m, u_latest, T)

    # Construct inequality matrix
    P = mpc_to_qp_ineq_mat(constraint_dict, n, m, T)

    # Construct inequality vector
    h = mpc_to_qp_ineq_vec(constraint_dict, u_latest, T)

    # Construct equality matrix
    C = mpc_to_qp_eq_mat(system_dict, n, m, T)

    # Construct equality vector
    b = mpc_to_qp_eq_vec(system_dict, x0, n, T)

    ret_dict = Dict("Q" => H, "q" => g, "G" => P, "h" => h, "A" => C, "b" => b)
    return ret_dict
end


"""
    mpc_to_qp_hessian(cost_dict, n, m, T) 

Construct QP Hessian. Refer to `mpc_to_qp` function for docstring.
"""
function mpc_to_qp_hessian(cost_dict, n, m, T) 
    H = zeros((T*(n+m), T*(n+m)))
    for idx in 1:T
        beg_u = (idx-1)*(n+m) + 1
        end_u = (idx-1)*(n+m) + 1 + (m-1)
        beg_x = (idx-1)*(n+m) + 1 + m
        end_x = idx*(n+m)

        # indices for off-diagonal blocks
        beg_u_next = idx*(n+m) + 1
        end_u_next = idx*(n+m) + 1 + (m-1)

        if idx < T
            H[beg_u:end_u, beg_u:end_u] = cost_dict["R"] + 2*cost_dict["S"]
            H[beg_x:end_x, beg_x:end_x] = cost_dict["Q"]

            # off-diagonal blocks
            H[beg_u:end_u, beg_u_next:end_u_next] = -cost_dict["S"]
            H[beg_u_next:end_u_next, beg_u:end_u] = -cost_dict["S"]
        else
            H[beg_u:end_u, beg_u:end_u] = cost_dict["R"] + cost_dict["S"]
            H[beg_x:end_x, beg_x:end_x] = cost_dict["Q_T"]
        end
    end
    return H
end


"""
    mpc_to_qp_ineq_mat(constraint_dict, n, m, T)

Construct inequality matrix. Refer to `mpc_to_qp` function for docstring.
"""
function mpc_to_qp_ineq_mat(constraint_dict, n, m, T) 
    Fu_size = size(constraint_dict["F_u"])[1]
    Fdu_size = size(constraint_dict["F_du"])[1]
    Fx_size = size(constraint_dict["F_x"])[1]
    FT_size = size(constraint_dict["F_T"])[1]
    row_block_size = Fu_size + Fdu_size + Fx_size
    P_rows = (T-1)*row_block_size + (Fu_size + Fdu_size + FT_size)

    P = zeros(P_rows, T*(n+m))
    for idx in 1:T
        beg_u_row = (idx-1)*row_block_size + 1
        end_u_row = (idx-1)*row_block_size + 1 + (Fu_size-1)
        beg_du_row = (idx-1)*row_block_size + 1 + Fu_size
        end_du_row = (idx-1)*row_block_size + 1 + Fu_size + (Fdu_size-1)
        beg_x_row = (idx-1)*row_block_size + 1 + Fu_size + Fdu_size

        if idx < T
            end_x_row = idx*row_block_size
        else
            end_x_row = (idx-1)*row_block_size + 1 + Fu_size + Fdu_size + (FT_size - 1)
        end

        beg_u_col = (idx-1)*(n+m) + 1
        end_u_col = (idx-1)*(n+m) + 1 + (m-1)
        beg_x_col = (idx-1)*(n+m) + 1 + m
        end_x_col = idx*(n+m)

        P[beg_u_row:end_u_row, beg_u_col:end_u_col] = constraint_dict["F_u"]
        P[beg_du_row:end_du_row, beg_u_col:end_u_col] = constraint_dict["F_du"]

        if idx < T
            P[beg_x_row:end_x_row, beg_x_col:end_x_col] = constraint_dict["F_x"]
        else
            P[beg_x_row:end_x_row, beg_x_col:end_x_col] = constraint_dict["F_T"]
        end

        if idx > 1
            beg_u_last_col = (idx-2)*(n+m) + 1
            end_u_last_col = (idx-2)*(n+m) + 1 + (m-1)
            P[beg_du_row:end_du_row, beg_u_last_col:end_u_last_col] = -constraint_dict["F_du"]
        end 
    end
    return P
end


"""
    mpc_to_qp_eq_mat(system_dict, n, m, T)

Construct equality matrix. Refer to `mpc_to_qp` function for docstring.
"""
function mpc_to_qp_eq_mat(system_dict, n, m, T)
    # Construct equality matrix
    C = zeros((T*n, T*(n+m)))
    for idx in 1:T
        beg_u_col = (idx-1)*(n+m) + 1
        end_u_col = (idx-1)*(n+m) + 1 + (m-1)
        beg_x_col = (idx-1)*(n+m) + 1 + m
        end_x_col = idx*(n+m)

        beg_x_row = (idx-1)*n + 1
        end_x_row = idx*n

        C[beg_x_row:end_x_row, beg_u_col:end_u_col] = -system_dict["B"]
        C[beg_x_row:end_x_row, beg_x_col:end_x_col] = Matrix{Float64}(I, n, n)

        if idx < T 
            beg_x_next_row = idx*n + 1
            end_x_next_row = (idx+1)*n
            C[beg_x_next_row:end_x_next_row, beg_x_col:end_x_col] = -system_dict["A"]
        end
    end
    return C
end


"""
    mpc_to_qp_linear_term(cost_dict, n, m, u_latest, T)

Construct linear term of QP cost. Refer to `mpc_to_qp` function for docstring.
"""
function mpc_to_qp_linear_term(cost_dict, n, m, u_latest, T)
    g = zeros((T*(n+m),))
    for idx in 1:T
        beg_u = (idx-1)*(n+m) + 1
        end_u = (idx-1)*(n+m) + 1 + (m-1)
        beg_x = (idx-1)*(n+m) + 1 + m
        end_x = idx*(n+m)

        if idx == 1
            g[beg_u:end_u] = -2*cost_dict["S"] * u_latest + cost_dict["r"]
        else
            g[beg_u:end_u] = cost_dict["r"]
        end

        if idx < T
            g[beg_x:end_x] = cost_dict["q"]
        else
            g[beg_x:end_x] = cost_dict["q_T"]
        end
    end
    return g
end


"""
    mpc_to_qp_ineq_vec(constraint_dict, u_latest, T)

Construct inequality vector. Refer to `mpc_to_qp` function for docstring.
"""
function mpc_to_qp_ineq_vec(constraint_dict, u_latest, T)
    Fu_size = size(constraint_dict["f_u"])[1]
    Fdu_size = size(constraint_dict["f_du"])[1]
    Fx_size = size(constraint_dict["f_x"])[1]
    FT_size = size(constraint_dict["f_T"])[1]
    row_block_size = Fu_size + Fdu_size + Fx_size
    h_rows = (T-1)*row_block_size + (Fu_size + Fdu_size + FT_size)

    h = zeros((h_rows,))

    for idx in 1:T
        beg_u = (idx-1)*row_block_size + 1
        end_u = (idx-1)*row_block_size + 1 + (Fu_size-1)
        beg_du = (idx-1)*row_block_size + 1 + Fu_size
        end_du = (idx-1)*row_block_size + 1 + Fu_size + (Fdu_size-1)
        beg_x = (idx-1)*row_block_size + 1 + Fu_size + Fdu_size

        h[beg_u:end_u] = constraint_dict["f_u"]
        h[beg_du:end_du] = constraint_dict["f_du"]

        if idx == 1
            h[beg_du:end_du] += constraint_dict["F_du"] * u_latest
        end

        if idx < T
            end_x = idx*row_block_size
            h[beg_x:end_x] = constraint_dict["f_x"]
        else
            end_x = (idx-1)*row_block_size + 1 + Fu_size + Fdu_size + (FT_size - 1)
            h[beg_x:end_x] = constraint_dict["f_T"]
        end
    end
    return h
end


"""
    mpc_to_qp_eq_vec(system_dict, x0, n, T)

Construct equality vector. Refer to `mpc_to_qp` function for docstring.
"""
function mpc_to_qp_eq_vec(system_dict, x0, n, T)
    b = zeros((T*n,))

    for idx in 1:T
        beg_x = (idx-1)*n + 1
        end_x = idx*n

        b[beg_x:end_x] = system_dict["w"]
        if idx == 1
            b[beg_x:end_x] += system_dict["A"] * x0
        end
    end
    return b
end


"""
    box_constraints(limit_dict)

Compute box constraints for x, u, and du.

# Arguments
- `limit_dict::Dict{String, Array}`: upper and lower limits on decision variables. Contains key-value pairs
        - "x_ub"::Array: the upper limit of the state variable x,
        - "x_lb"::Array: the lower limit of the state variable x,
        - "u_ub"::Array: the upper limit of the input variable u,
        - "u_lb"::Array: the lower limit of the input variable u,
        - "du_ub"::Array: the upper limit of the differenced input variable du,
        - "du_lb"::Array: the lower limit of the differenced input variable du,
        - "x_T_ub"::Array: the upper limit of the state variable x,
        - "x_T_lb"::Array: the lower limit of the state variable x.

# Returns
- `Dict{String, Array}`: inequality constraint parameters of the quadratic program. Contains key-value pairs
        - "F_x"::Array: the matrix of the inequality constraint for x,
        - "f_x"::Array: the vector of the inequality constraint for x,
        - "F_u"::Array: the matrix of the inequality constraint for u,
        - "f_u"::Array: the vector of the inequality constraint for u,
        - "F_du"::Array: the matrix of the inequality constraint for du,
        - "f_du"::Array: the vector of the inequality constraint for du,
        - "F_T"::Array: the matrix of the inequality constraint for the terminal state,
        - "f_T"::Array: the vector of the inequality constraint for the terminal state.
"""
function box_constraints(limit_dict)
    n = size(limit_dict["x_ub"])[1]
    m = size(limit_dict["u_ub"])[1]

    F_x = vcat(Matrix{Float64}(I, n, n), -Matrix{Float64}(I, n, n))
    f_x = vcat(limit_dict["x_ub"], -limit_dict["x_lb"])

    F_u = vcat(Matrix{Float64}(I, m, m), -Matrix{Float64}(I, m, m))
    f_u = vcat(limit_dict["u_ub"], -limit_dict["u_lb"])

    F_du = vcat(Matrix{Float64}(I, m, m), -Matrix{Float64}(I, m, m))
    f_du = vcat(limit_dict["du_ub"], -limit_dict["du_lb"])

    F_T = vcat(Matrix{Float64}(I, n, n), -Matrix{Float64}(I, n, n))
    f_T = vcat(limit_dict["x_T_ub"], -limit_dict["x_T_lb"])
    
    return Dict("F_x" => F_x, "f_x" => f_x, "F_u" => F_u, "f_u" => f_u, "F_du" => F_du, "f_du" => f_du, "F_T" => F_T, "f_T" => f_T)
end


"""
    separate_solution(sol, n, m, u_latest, T)

Separate the solution output by an optimal control problem into three arrays containing
the states, the inputs, and the input differences.

# Arguments
- `sol::Array`: the solution output by the QP.
- `n:Integer64`: the state dimension.
- `m:Integer64`: the input dimension.
- `u_latest:Array`: the last input before the optimal control problem was solved.
- `T:Integer64`: the prediction horizon.

# Returns
- `Dict{String, Array}`: three arrays with keys "x", "u", and "du" of size T*n, T*m, and T*m respectively.
"""
function separate_solution(sol, n, m, u_latest, T)
    x = zeros((T*n,))
    u = zeros((T*m,))
    du = zeros((T*m,))

    for idx in 1:T
        beg_x_qp = (idx-1)*(n+m) + 1 + m
        end_x_qp = idx*(n+m)
        beg_u_qp = (idx-1)*(n+m) + 1
        end_u_qp = (idx-1)*(n+m) + 1 + (m-1)

        beg_x = (idx-1)*n + 1
        end_x = idx*n
        x[beg_x:end_x] = sol[beg_x_qp:end_x_qp]

        beg_u = (idx-1)*m + 1
        end_u = idx*m
        u[beg_u:end_u] = sol[beg_u_qp:end_u_qp]
        du[beg_u:end_u] = sol[beg_u_qp:end_u_qp] - u_latest
        
        u_latest = u[beg_u:end_u]
    end

    return Dict("x" => x, "u" => u, "du" => du)
end


"""
    nonlinear_eq_constraint(z, dyn)

Given the primal variable of an NLP problem and a (nonlinear) dynamics function,
output the vector representing the equality constraints in the NLP.

# Arguments
- `z::Array`: the primal variable in an NLP problem.
- `x0::Array`: the initial state.
- `n_x::Integer64`: the state dimension.
- `n_u::Integer64`: the input dimension.
- `T::Integer64`: the time horizon of the optimal control problem.
- `dyn:Function`: the dynamics function.
- `constraint_vec:Array`: allocated memory for a previously computed equality constraint.
        If equals to nothing, allocate new memory for vector.

# Returns
- `Array`: the vector encoding equality constraints.
"""
function nonlinear_eq_constraint(z, x0, n_x, n_u, T, dyn, constraint_vec=nothing)
    
    new_vec = false
    if isnothing(constraint_vec)
        new_vec = true
        constraint_vec = zeros((T*n_x,))
    end

    last_x = x0
    for idx in 1:T
        beg_u_z = (idx-1)*(n_x + n_u) + 1
        end_u_z = (idx-1)*(n_x + n_u) + 1 + (n_u-1)
        beg_x_z = (idx-1)*(n_x + n_u) + 1 + n_u
        end_x_z = idx*(n_x + n_u)
        
        last_u = z[beg_u_z:end_u_z]
        curr_x = z[beg_x_z:end_x_z]

        beg_x_const = (idx-1)*n_x + 1
        end_x_const = idx*n_x
        constraint_vec[beg_x_const:end_x_const] = curr_x - dyn(last_x, last_u)

        last_x = curr_x
    end

    if new_vec
        return constraint_vec
    end
end


"""
    nonlinear_eq_jacobian(z, x0, n_x, n_u, T, f_x, f_u, jac=nothing)

Compute the Jacobian of the nonlinear inequality vector.

# Arguments
- `z::Array`: the primal variable in an NLP problem.
- `x0::Array`: the initial state.
- `n_x::Integer64`: the state dimension.
- `n_u::Integer64`: the input dimension.
- `T::Integer64`: the time horizon of the optimal control problem.
- `f_x:Function`: the Jacobian of the dynamics function with respect to state.
- `f_u:Function`: the Jacobian of the dynamics function with respect to input.
- `constraint_vec:Array`: allocated memory for a previously computed equality constraint.
        If equals to nothing, allocate new memory for vector.

# Returns
- `Array`: the Jacobian.
"""
function nonlinear_eq_jacobian(z, x0, n_x, n_u, T, f_x, f_u, jac=nothing)
    new_jac = false

    if isnothing(jac)
        new_jac = true
        jac = zeros((T*n_x, T*(n_x + n_u)))
    end

    x_curr = x0
    u_curr = z[1:n_u]
    for idx in 1:T
        beg_u_col = (idx-1)*(n_x + n_u) + 1
        end_u_col = (idx-1)*(n_x + n_u) + 1 + (n_u-1)
        beg_x_col = (idx-1)*(n_x + n_u) + 1 + n_u
        end_x_col = idx*(n_x + n_u)

        beg_x_row = (idx-1)*n_x + 1
        end_x_row = idx*n_x

        jac[beg_x_row:end_x_row, beg_u_col:end_u_col] = -f_u(x_curr, u_curr)
        if new_jac
            jac[beg_x_row:end_x_row, beg_x_col:end_x_col] = Matrix{Float64}(I, n_x, n_x)
        end
        
        if idx < T 
            x_curr = z[beg_x_col:end_x_col]
            
            beg_u_next_col = idx*(n_x + n_u) + 1
            end_u_next_col = idx*(n_x + n_u) + 1 + (n_u-1)
            u_curr = z[beg_u_next_col:end_u_next_col]

            beg_x_next_row = idx*n_x + 1
            end_x_next_row = (idx+1)*n_x
            jac[beg_x_next_row:end_x_next_row, beg_x_col:end_x_col] = -f_x(x_curr, u_curr)
        end
    end

    if new_jac
        return jac
    end
end
