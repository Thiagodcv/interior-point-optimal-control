
"""
    pdip_qp(param_dict, x0)

Primal-dual interior point QP with Mehrotra correction.

# Arguments
- `param_dict::Dict{String, Array}`: the parameters which define the QP problem. Contains key-value pairs
        - "Q"::Array: the Hessian of the QP (n,n),
        - "q"::Array: the linear term of the QP (n,),
        - "G"::Array: the linear inequality constraint matrix (p,n),
        - "h"::Array: the linear inequality constraint limits (p,),
        - "A"::Array: the linear equality constraint matrix (m,n),
        - "b"::Array: the linear equality constraint limits (m,).
- `x0::Array`: the initial primal iterate (n,). Must satisfy inequality constraints.

# Returns
- `Dict{String, Array}`: primal dual solution and optimization details.
"""
function pdip_qp(param_dict, x0)
    epsilon_feas = 1e-6
    epsilon = 1e-6

    max_iters = 100

    # Find dimensions
    n = size(x0)[1]
    p = size(param_dict["G"])[1]
    m = size(param_dict["A"])[1]

    # Set initial primal and dual iterates
    x = x0
    lambda = -1 ./ (param_dict["G"] * x0 - param_dict["h"])
    nu = ones((m,))  # Is this a good initial guess for nu?
end
