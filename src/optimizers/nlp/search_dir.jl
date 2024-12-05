using LinearAlgebra


"""
    compute_kkt_residual(z, lambda, nu, s, param, res_mat)

Computes the residual vector which encodes the KKT conditions of the NLP program.

NOTE: does NOT include the centering parameter nor the Mehrotra correction. Those are added on
by different functions later.

NOTE: this function also assumes the KKT matrix is made symmetrical.

# Arguments
- `z::Array`: the current primal iterate.
- `lambda::Array`: the current dual iterate for the inequality constraint. Strictly positive.
- `nu::Array`: the current dual iterate for the equality constraint.
- `s::Array`: the current slack variable iterate for the inequality constraint. Strictly positive.
- `param::Dict{String, Array}`: the parameters which define the QP problem. Contains key-value pairs
        - "H"::Array: the Hessian of the NLP,
        - "g"::Array: the linear term of the NLP,
        - "P"::Array: the inequality constraint matrix,
        - "h"::Array: the inequality constraint vector,
        - "eq_vec"::Array: the equality constraint residual vector evaluated at z.
        - "eq_jac"::Array: the Jacobian of the equality constraint residual vector evaluated at z.
- `res_mat::Array`: an (n+2p+m,) array which is overwritten, and transformed into the KKT residual.

# Returns
- `Array`: the residual vector. Only returns if argument for res_mat not specified by user.
"""
function compute_kkt_residual(z, lambda, nu, s, param, res_mat=nothing)
    n_z = size(z)[1]
    n_lam = size(lambda)[1]
    n_nu = size(nu)[1]

    new_res_mat = false
    if isnothing(res_mat)
        new_res_mat = true
        res_mat = zeros((n_z + 2*n_lam + n_nu,))
    end

    res_mat[1:n_z] = param["H"]*z + param["g"] + transpose(param["P"])*lambda + transpose(param["eq_jac"])*nu
    res_mat[n_z+1:n+lam] = lambda  # Assumes the KKT matrix is made symmetric.
    res_mat[n_z+n_lam+1:n_z+2*n_lam] = param["P"]*z - param["h"] + s
    res_mat[n_z+2*n_lam+1:n_z+2*n_lam+n_nu] = param["eq_vec"]

    if new_res_mat
        return res_mat
    end
end

