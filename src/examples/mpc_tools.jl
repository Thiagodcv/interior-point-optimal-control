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

    # Construct inequality matrix
    P = mpc_to_qp_ineq_mat(constraint_dict, n, m, T)

    # Construct equality matrix
    C = mpc_to_qp_eq_mat(system_dict, n, m, T)

    # Construct linear term of QP cost
    g = mpc_to_qp_linear_term(cost_dict, n, m, T)
end


function mpc_to_qp_hessian(cost_dict, n, m, T) 
    # Construct QP Hessian
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


function mpc_to_qp_ineq_mat(constraint_dict, n, m, T)
    # Construct inequality matrix 
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

function mpc_to_qp_linear_term(cost_dict, n, m, T)
    # Construct linear term of QP cost
    g = zeros((T*(n+m),))
    for idx in 1:T
        beg_u = (idx-1)*(n+m) + 1
        end_u = (idx-1)*(n+m) + 1 + (m-1)
        beg_x = (idx-1)*(n+m) + 1 + m
        end_x = idx*(n+m)

        g[beg_u:end_u] = -2*cost_dict["S"] * u_latest + cost_dict["r"]

        if idx < T
            g[beg_x:end_x] = cost_dict["q"]
        else
            g[beg_x:end_x] = cost_dict["q_T"]
        end
    end
    return g
end
