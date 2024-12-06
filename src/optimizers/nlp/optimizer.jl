
"""
    pdip_nlp(param_dict, x0)

Primal-dual interior point NLP with Mehrotra correction. This implementation
is specifically tailored to optimal control problems.

# Arguments
- `params::Dict{String, Array}`: the parameters of the NLP problem. Contains key-value pairs
        - "H"::Array: the Hessian of the NLP,
        - "g"::Array: the linear term of the NLP,
        - "P"::Array: the inequality constraint matrix,
        - "h"::Array: the inequality constraint vector
- `z0::Array`: the initial primal iterate. Must satisfy inequality constraints. 

# Returns
- `Dict{String, Array}`: primal dual solution and optimization details.
"""
function pdip_nlp(param_dict, z0)
    epsilon = 1e-7
    max_iters = 100

    n_z = size(z0)[1]
    n_lam = size(P)[1]
end
