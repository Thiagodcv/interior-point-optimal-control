using LinearAlgebra


"""
    compute_kkt_residual(param_dict, x, lambda, nu, s, res_mat)

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
- `x::Array`: the current primal iterate (n,).
- `lambda::Array`: the current dual iterate for the inequality constraint (p,). Strictly positive.
- `nu::Array`: the current dual iterate for the equality constraint (m,).
- `s::Array`: the current slack variable iterate for the inequality constraint (p,). Strictly positive.
- `res_mat::Array`: an (n+2p+m,) array which is overwritten, and transformed into the KKT residual.

# Returns
- `Array`: the residual vector. Only returns if argument for res_mat not specified by user.
"""
function compute_kkt_residual(param_dict, x, lambda, nu, s, res_mat=nothing)
    n = size(param_dict["Q"])[1]
    p = size(param_dict["G"])[1]
    m = size(param_dict["A"])[1]

    new_res_mat = false
    if isnothing(res_mat)
        new_res_mat = true
        res_mat = zeros((n+2*p+m,))
    end

    res_mat[1:n] = param_dict["Q"] * x + param_dict["q"] + transpose(param_dict["G"]) * lambda + transpose(param_dict["A"]) * nu
    res_mat[n+1:n+p] = Diagonal(s) * lambda
    res_mat[n+p+1:n+2*p] = param_dict["G"] * x + s - param_dict["h"]
    res_mat[n+2*p+1:n+2*p+m] = param_dict["A"] * x - param_dict["b"]

    if new_res_mat
        return res_mat
    end
end


"""
    compute_kkt_jacobian(param_dict, lambda, s)

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
- `jac::Array`: an (n+2p+m,n+2p+m) array which was output by this very function, but which will be updated with new `lambda` and `s` iterates. 
                all other entries left untouched.

# Returns
- `Array`: the Jacobian matrix. Only returns if argument for jac not specified by user.
"""
function compute_kkt_jacobian(param_dict, lambda, s, jac=nothing)
    n = size(param_dict["Q"])[1]
    p = size(param_dict["G"])[1]
    m = size(param_dict["A"])[1]

    new_jac = false
    if isnothing(jac)
        new_jac = true
        jac = zeros((n + 2*p + m, n + 2*p + m))
    
        # First row of block matrices
        jac[1:n, 1:n] = param_dict["Q"]
        jac[1:n, n+p+1:n+2*p] = transpose(param_dict["G"]) 
        jac[1:n, n+2*p+1:n+2*p+m] = transpose(param_dict["A"]) 

        # Third row of block matrices
        jac[n+p+1:n+2*p, 1:n] = param_dict["G"]
        jac[n+p+1:n+2*p, n+1:n+p] = Matrix{Float64}(I, p, p)

        # Fourth row of block matrices
        jac[n+2*p+1:n+2*p+m, 1:n] = param_dict["A"]
    end

    # Second row of block matrices. Left for last as these change per iteration.
    jac[n+1:n+p, n+1:n+p] = Diagonal(lambda)
    jac[n+1:n+p, n+p+1:n+2*p] = Diagonal(s)

    if new_jac
        return jac
    end
end


"""
    compute_affine_scaling_dir(kkt_res, kkt_jac, n, p, m)

Compute affine scaling steps for the primal and dual iterates. 

# Arguments
- `kkt_res::Array`: the residual vector (n+2p+m,) which encodes the KKT conditions of a quadratic program. Does NOT include centering / Mehrotra correction.
- `kkt_jac::Array`: the (n+2p+m,n+2p+m) Jacobian of the residual vector described above.
- `n::Integer64`: size of the primal variable x.
- `p::Integer64`: number of inequality constraints.
- `m::Integer64`: number of equality constraints.

# Returns
- `Dict{String, Array}`: the affine scaling steps for each primal and dual variable. 
"""
function compute_affine_scaling_dir(kkt_res, kkt_jac, n, p, m)
    aff_step = kkt_jac \ (-kkt_res)
    d_x_aff = aff_step[1:n]
    d_s_aff = aff_step[n+1:n+p]
    d_lambda_aff = aff_step[n+p+1:n+2*p]
    d_nu_aff = aff_step[n+2*p+1:n+2*p+m]

    return Dict("x" => d_x_aff, "s" => d_s_aff, "lambda" => d_lambda_aff, "nu" => d_nu_aff)
end


"""
    compute_centering_plus_corrector_dir(kkt_jac, d_s_aff, d_lambda_aff, sigma, mu, n, p, m)

Compute centering-plus-corrector directions. NOTE: LOOK INTO REUSING CC_VEC.

# Arguments
- `kkt_jac::Array`: the (n+2p+m,n+2p+m) Jacobian of the residual vector described above.
- `d_s_aff::Array`: the affine scaling direction for the slack vector (p,) associated with the inequality constraint.
- `d_p_aff::Array`: the affine scaling direction for the dual vector (p,) associated with the inequality constraint.
- `n::Integer64`: size of the primal variable x.
- `p::Integer64`: number of inequality constraints.
- `m::Integer64`: number of equality constraints.

# Returns
- `Dict{String, Array}`: the centering-plus-corrector directions for each primal and dual variable. 
"""
function compute_centering_plus_corrector_dir(kkt_jac, d_s_aff, d_lambda_aff, sigma, mu, n, p, m)
    cc_vec = zeros((n+2*p+m,))
    cc_vec[n+1:n+p] = sigma * mu - Diagonal(d_s_aff) * d_lambda_aff
    cc_step = kkt_jac \ cc_vec
    d_x_cc = cc_step[1:n]
    d_s_cc = cc_step[n+1:n+p]
    d_lambda_cc = cc_step[n+p+1:n+2*p]
    d_nu_cc = cc_step[n+2*p+1:n+2*p+m]
    return Dict("x" => d_x_cc, "s" => d_s_cc, "lambda" => d_lambda_cc, "nu" => d_nu_cc)
end
