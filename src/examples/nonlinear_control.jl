include("../tools/mpc_tools.jl")
include("../optimizers/nlp/optimizer2.jl")
using Plots


# initial state, the latest input, and the time horizon
x0 = [0.6*pi; 2.]
u_latest = [0.]
T = 20
n_x = 2
n_u = 1

# Cost parameters
d = [pi; 0]
c = 1.
cost_dict = Dict()
cost_dict["Q"] = c*Matrix{Float64}(I, n_x, n_x)
cost_dict["q"] = -2*c*d
cost_dict["R"] = 0.01*Matrix{Float64}(I, n_u, n_u)
cost_dict["r"] = zeros((n_u,))
cost_dict["S"] = cost_dict["R"]
cost_dict["Q_T"] = cost_dict["Q"]
cost_dict["q_T"] = cost_dict["q"]

# Parameters for variable limits
big_num = 10_000
limit_dict = Dict()
limit_dict["x_ub"] = [big_num; big_num]
limit_dict["x_lb"] = -limit_dict["x_ub"]

limit_dict["u_ub"] = [big_num]
limit_dict["u_lb"] = -limit_dict["u_ub"]

limit_dict["du_ub"] = [2.]
limit_dict["du_lb"] = -limit_dict["du_ub"]

limit_dict["x_T_ub"] = [big_num; big_num]
limit_dict["x_T_lb"] = -limit_dict["x_T_ub"]
constraint_dict = box_constraints(limit_dict)

# Parameters for LTI system
system_dict = Dict()
system_dict["A"] = zeros((2,2))
system_dict["B"] = zeros((2,))
system_dict["w"] = zeros((2,))

qp_dict = mpc_to_qp(cost_dict, constraint_dict, system_dict, x0, u_latest, T)

param = Dict()
param["H"] = qp_dict["Q"]  # Usually would have to multiply by 2 but taken care of by function.
param["g"] = qp_dict["q"]
param["P"] = qp_dict["G"]
param["h"] = qp_dict["h"]

g = 9.8  # nlp diverges for higher than 9.835
dt = 0.4

# The continuous-time dynamics function
function f(x, u)
    return [x[2]; -g*sin(x[1]) - x[2] + u[1]]
end

# Jacobian of f w.r.t. x
function f_x(x, u)
    return [0. 1.; -g*cos(x[1]) (-1.)]
end

# Jacobian of f w.r.t. u
function f_u(x, u)
    return [0.; 1.]
end

# The discrete-time dynamics function
function h(x, u)
    return x + f(x, u)*dt
end

# Jacobian of h w.r.t. x
function h_x(x, u)
    return [1. 0.; 0. 1.] + f_x(x, u)*dt
end

# Jacobian of h w.r.t. u
function h_u(x, u)
    return f_u(x, u)*dt
end

# constraint_vec = zeros((T*n_x,))
# jac = zeros((T*n_x, T*(n_x + n_u)))

function eq_vec(z)
    return nonlinear_eq_constraint(z, x0, n_x, n_u, T, h)
end

function eq_jac(z)
    return nonlinear_eq_jacobian(z, x0, n_x, n_u, T, h_x, h_u)
end

eq_consts = Dict("vec" => eq_vec, "jac" => eq_jac)
# z_init = zeros((T*(n_x+n_u),))
z_init = repeat([0.; pi; 0.], T)

ret = pdip_nlp(param, eq_consts, z_init)
ret_separate = separate_solution(ret["z"], n_x, n_u, u_latest, T)

# println("iters: ", ret["iters"])
# # println("solution: ", ret["x"])
# println("state solution: ", ret_separate["x"])
# println("input solution: ", ret_separate["u"])
# println("diff input solution: ", ret_separate["du"])

# z = ret["z"]
# println("Objective at solution: ", z' * param["H"] * z + param["g"]' * z)
# println("Equality constraints at solution: ", norm(eq_vec(z)))
# z_test = repeat([0.; pi; 0.], T)
# println("Objective at test: ", z_test' * param["H"] * z_test + param["g"]' * z_test)
# println("Equality constraints at test: ", norm(eq_vec(z_test)))

x1 = vcat(x0[1], ret_separate["x"][1:2:end])
x2 = vcat(x0[2], ret_separate["x"][2:2:end])
u = vcat(ret_separate["u"], missing)
du = vcat(ret_separate["du"], missing)
t_steps = dt*collect(0:T)

yticks1 = ([1.8, 2.6, 3.4], ["1.8", "2.6", "3.4"])
p1 = plot(t_steps, x1, ylabel="θₜ", ylim=(1.8, 3.4), yticks=yticks1, legend=false)
hline!(p1, [pi], color=:red, linewidth=0.5)

yticks2 = ([-1., -0., 1., 2.], ["-1.0", "0.0", "1.0", "2.0"])
p2 = plot(t_steps, x2, ylabel="θ'ₜ", ylim=(-1, 2), yticks=yticks2, legend=false)
hline!(p2, [0.], color=:red, linewidth=0.5)

yticks3 = ([0., 3., 6., 9.], ["0.0", "3.0", "6.0", "9.0"])
p3 = plot(t_steps, u, ylabel="uₜ", ylim=(-2, 10), yticks=yticks3, legend=false)
hline!(p3, [0.], color=:red, linewidth=0.5)

p4 = plot(t_steps, du, ylabel="Δuₜ", xlabel="t (seconds)", ylim=(-2.5, 2.5), legend=false)
hline!(p4, [0.], color=:red, linewidth=0.5)
hline!(p4, [2.], color=:orange, linewidth=1, linestyle=:dash)
hline!(p4, [-2.], color=:orange, linewidth=1, linestyle=:dash)

plot(p1, p2, p3, p4, layout=(4,1))

# combined = plot(p1, p2, p3, p4, layout=(4,1))
# png(combined, "nlp.png")
