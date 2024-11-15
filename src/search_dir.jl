using LinearAlgebra


"""
    compute_kkt_residual(param_dict, res_array, x, lambda, nu, s)

Computes the residual vector which encodes the KKT conditions of a quadratic program.
NOTE: does NOT include the centering parameter nor the Mehrotra correction. Those are added on
by different functions later.

# Arguments
- `param_dict::Dict{String, Array}`: the parameters which define the QP problem. Contains key-value pairs
        - "Q"::Array: the Hessian of the QP (n,n),
        - "q"::Array: the linear term of the QP (n,),
        - "G"::Array: the linear inequality constraint matrix (p,n),
        - "h"::Array: the linear inequality constraint limits (p,),
        - "A"::Array: the linear equality constraint matrix (m,n),
        - "b"::Array: the linear equality constraint limits (m,).
- `res_array::Array`: an (n+2p+m,) array which is overwritten, and transformed into the KKT residual.
- `x::Array`: the current primal iterate (n,).
- `lambda::Array`: the current dual iterate for the inequality constraint (p,). Strictly positive.
- `nu::Array`: the current dual iterate for the equality constraint (m,).
- `s::Array`: the current slack variable iterate for the inequality constraint (p,). Strictly positive.
"""
function compute_kkt_residual(param_dict, res_array, x, lambda, nu, s)
    n = size(param_dict["Q"])[1]
    p = size(param_dict["G"])[1]
    m = size(param_dict["A"])[1]

    res_array[1:n] = param_dict["Q"] * x + param_dict["q"] + transpose(param_dict["G"]) * lambda + transpose(param_dict["A"]) * nu
    res_array[n+1:n+p] = Diagonal(s) * lambda
    res_array[n+p+1:n+2*p] = param_dict["G"] * x + s - param_dict["h"]
    res_array[n+2*p+1:n+2*p+m] = param_dict["A"] * x - param_dict["b"]
end


"""
    compute_kkt_Jacobian(param_dict, lambda, s)

Computes the Jacobian of the residual vector which encodes the KKT conditions of a quadratic program.
NOTE: does NOT include the centering parameter.

# Arguments
- `param_dict::Dict{String, Array}`: the parameters which define the QP problem. Contains key-value pairs
        - "Q"::Array: the Hessian of the QP (n,n),
        - "q"::Array: the linear term of the QP (n,),
        - "G"::Array: the linear inequality constraint matrix (p,n),
        - "h"::Array: the linear inequality constraint limits (p,),
        - "A"::Array: the linear equality constraint matrix (m,n),
        - "b"::Array: the linear equality constraint limits (m,).
- `lambda::Array`: the current dual iterate for the inequality constraint (p,). Strictly positive.
- `s::Array`: the current slack variable iterate for the inequality constraint (p,). Strictly positive.
"""
function compute_kkt_Jacobian(param_dict, lambda, s)
    n = size(param_dict["Q"])[1]
    p = size(param_dict["G"])[1]
    m = size(param_dict["A"])[1]
    jac = zeros((n + 2*p + m, n + 2*p + m))

    # First row of block matrices
    jac[1:n, 1:n] = param_dict["Q"]
    jac[1:n, n+p+1:n+2*p] = transpose(param_dict["G"]) 
    jac[1:n, n+2*p+1:n+2*p+m] = transpose(param_dict["A"]) 

    # Second row of block matrices
    jac[n+1:n+p, n+1:n+p] = Diagonal(lambda)
    jac[n+1:n+p, n+p+1:n+2*p] = Diagonal(s)

    # Third row of block matrices
    jac[n+p+1:n+2*p, 1:n] = param_dict["G"]
    jac[n+p+1:n+2*p, n+1:n+p] = Matrix{Float64}(I, p, p)

    # Fourth row of block matrices
    jac[n+2*p+1:n+2*p+m, 1:n] = param_dict["A"]

    return jac
end

function compute_affine_scaling_dir()
    return -1
end

function compute_centering_plus_corrector_dir()
    return -1
end
