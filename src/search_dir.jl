using LinearAlgebra


"""
    compute_kkt_residual(param_dict, res_array, x, s, lambda, nu)

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
- `s::Array`: the current slack variable iterate for the inequality constraint (p,). Strictly positive.
- `lambda::Array`: the current dual iterate for the inequality constraint (p,). Strictly positive.
- `nu::Array`: the current dual iterate for the equality constraint (m,).

# Returns
- `Array`: the KKT residual. OR MAYBE DON'T RETURN -- JUST MODIFY RES_ARRAY
"""
function compute_kkt_residual(param_dict, res_array, x, s, lambda, nu)
    n = size(param_dict["Q"])[1]
    p = size(param_dict["G"])[1]
    m = size(param_dict["A"])[1]

    res_array[1:n] = param_dict["Q"] * x + param_dict["q"] + transpose(param_dict["G"]) * lambda + transpose(param_dict["A"]) * nu
    res_array[n+1:n+p] = Diagonal(s) * lambda
    res_array[n+p+1:n+2*p] = param_dict["G"] * x + s - param_dict["h"]
    res_array[n+2*p+1:n+2*p+m] = param_dict["A"] * x - param_dict["b"]
end

function compute_kkt_Jacobian()
    return -1
end

function compute_affine_scaling_dir()
    return -1
end

function compute_centering_plus_corrector_dir()
    return -1
end
