include("../tools/mpc_tools.jl")
include("../optimizers/qp/optimizer.jl")


n = 2
m = 1

d = [0; 0]

# Cost parameters
cost_dict = Dict()
cost_dict["Q"] = Matrix{Float64}(I, n, n)
cost_dict["q"] = -2*d
cost_dict["R"] = 0.01*Matrix{Float64}(I, m, m)
cost_dict["r"] = zeros((m,))
cost_dict["S"] = cost_dict["R"]
cost_dict["Q_T"] = cost_dict["Q"]
cost_dict["q_T"] = cost_dict["q"]

# Parameters for variable limits
big_num = 10_000
limit_dict = Dict()
limit_dict["x_ub"] = [5.; 2.]
limit_dict["x_lb"] = -limit_dict["x_ub"]

limit_dict["u_ub"] = [1.]
limit_dict["u_lb"] = -limit_dict["u_ub"]

limit_dict["du_ub"] = [0.5]
limit_dict["du_lb"] = -limit_dict["du_ub"]

limit_dict["x_T_ub"] = [5.; 2.]
limit_dict["x_T_lb"] = -limit_dict["x_T_ub"]
constraint_dict = box_constraints(limit_dict)

# Parameters for LTI system
system_dict = Dict()
freq = 0.5
system_dict["A"] = [1. freq;
                    0. 1.]
system_dict["B"] = [0.; freq]
system_dict["w"] = zeros((n,))

# initial state, the latest input, and the time horizon
x0 = [4.; 1.]
u_latest = [0.]
T = 20

qp_dict = mpc_to_qp(cost_dict, constraint_dict, system_dict, x0, u_latest, T)

x_init = zeros((T*(n+m),))
ret = pdip_qp(qp_dict, x_init)
ret_separate = separate_solution(ret["x"], n, m, u_latest, T)
println("iters: ", ret["iters"])
# println("solution: ", ret["x"])
println("state solution: ", ret_separate["x"])
println("input solution: ", ret_separate["u"])
println("diff input solution: ", ret_separate["du"])
