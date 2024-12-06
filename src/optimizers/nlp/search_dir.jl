using LinearAlgebra


"""
    kkt_residual_nlp(z, lambda, nu, s, param, res_mat)

Computes the residual vector which encodes the KKT conditions of the NLP.

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
function kkt_residual_nlp(z, lambda, nu, s, param, res_mat=nothing)
    n_z = size(z)[1]
    n_lam = size(lambda)[1]
    n_nu = size(nu)[1]

    new_res_mat = false
    if isnothing(res_mat)
        new_res_mat = true
        res_mat = zeros((n_z + 2*n_lam + n_nu,))
    end

    res_mat[1:n_z] = param["H"]*z + param["g"] + transpose(param["P"])*lambda + transpose(param["eq_jac"])*nu
    res_mat[n_z+1:n_z+n_lam] = lambda  # Assumes the KKT matrix is made symmetric.
    res_mat[n_z+n_lam+1:n_z+2*n_lam] = param["P"]*z - param["h"] + s
    res_mat[n_z+2*n_lam+1:n_z+2*n_lam+n_nu] = param["eq_vec"]

    if new_res_mat
        return res_mat
    end
end


"""
    kkt_jacobian_nlp(lambda, s, B, param, jac)

Computes the Jacobian of the residual vector which encodes the KKT conditions of the NLP.

NOTE: does NOT include the centering parameter.
NOTE: matrix is assumed to have been made symmetrical by multipling the second block row by inverse(S).

# Arguments
- `lambda::Array`: the current dual iterate for the inequality constraint. Strictly positive.
- `s::Array`: the current slack variable iterate for the inequality constraint. Strictly positive.
- `B::Array`: (an approximation of) the Hessian of the Lagrangian.
- `param::Dict{String, Array}`: the parameters which define the QP problem. Contains key-value pairs
        - "P"::Array: the inequality constraint matrix,
        - "eq_jac"::Array: the Jacobian of the equality constraint residual vector evaluated at z.
- `jac::Array`: an array which was output by this very function, but which will be updated with new iterates.

# Returns
- `Array`: the Jacobian matrix. Only returns if argument for jac not specified by user.
"""
function kkt_jacobian_nlp(lambda, s, B, param, jac=nothing)
    n_z = size(B)[1]
    n_lam = size(lambda)[1]
    n_nu = size(param["eq_jac"])[1]

    new_jac = false
    if isnothing(jac)
        new_jac = true
        jac = zeros((n_z + 2*n_lam + n_nu, n_z + 2*n_lam + n_nu))
    
        # First row of block matrices
        jac[1:n_z, n_z+n_lam+1:n_z+2*n_lam] = transpose(param["P"]) 

        # Second row of block matrices
        jac[n_z+1:n_z+n_lam, n_z+n_lam+1:n_z+2*n_lam] = Matrix{Float64}(I, n_lam, n_lam)

        # Third row of block matrices
        jac[n_z+n_lam+1:n_z+2*n_lam, 1:n_z] = param["P"]
        jac[n_z+n_lam+1:n_z+2*n_lam, n_z+1:n_z+n_lam] = Matrix{Float64}(I, n_lam, n_lam)
    end

    # Left for last as these change per iteration:

    # First row of block matrices
    jac[1:n_z, 1:n_z] = B
    jac[1:n_z, n_z+2*n_lam+1:n_z+2*n_lam+n_nu] = transpose(param["eq_jac"]) 

    # Second row of block matrices. 
    jac[n_z+1:n_z+n_lam, n_z+1:n_z+n_lam] = Diagonal(1 ./ s) * Diagonal(lambda)
    
    # Fourth row of block matrices
    jac[n_z+2*n_lam+1:n_z+2*n_lam+n_nu, 1:n_z] = param["eq_jac"]

    if new_jac
        return jac
    end
end


"""
    centering_plus_corrector_dir_nlp(kkt_jac, d_s_aff, d_lambda_aff, sigma, s, mu, n, p, m)

Compute centering-plus-corrector directions. 

NOTE: Assumes KKT matrix has been made symmetric.

# Arguments
- `kkt_jac::Array`: the (n+2p+m,n+2p+m) Jacobian of the residual vector described above.
- `d_s_aff::Array`: the affine scaling direction for the slack vector (p,) associated with the inequality constraint.
- `d_p_aff::Array`: the affine scaling direction for the dual vector (p,) associated with the inequality constraint.
- `s::Array`: the slack variable.
- `mu::Float64`: the barrier parameter.
- `n::Integer64`: size of the primal variable x.
- `p::Integer64`: number of inequality constraints.
- `m::Integer64`: number of equality constraints.

# Returns
- `Dict{String, Array}`: the centering-plus-corrector directions for each primal and dual variable. 
"""
function centering_plus_corrector_dir_nlp(kkt_jac, d_s_aff, d_lambda_aff, sigma, s, mu, n, p, m)
    cc_vec = zeros((n+2*p+m,))
    cc_vec[n+1:n+p] = Diagonal(1 ./ s) * (sigma * mu * ones((p,)) - Diagonal(d_s_aff) * d_lambda_aff)
    cc_step = kkt_jac \ cc_vec
    d_x_cc = cc_step[1:n]
    d_s_cc = cc_step[n+1:n+p]
    d_lambda_cc = cc_step[n+p+1:n+2*p]
    d_nu_cc = cc_step[n+2*p+1:n+2*p+m]
    return Dict("x" => d_x_cc, "s" => d_s_cc, "lambda" => d_lambda_cc, "nu" => d_nu_cc)
end
