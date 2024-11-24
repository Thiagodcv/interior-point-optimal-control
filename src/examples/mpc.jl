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
    return -1
end
