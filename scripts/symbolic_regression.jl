#=============================================================
SCRIPT TO RECOVER THE FUNCTIONAL FORM OF THE TRANSMISSION RATE
=============================================================#
using Pkg
# Activate the project
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()
cd(@__DIR__)
using DrWatson
@quickactivate("UDE_FUNCTIONAL_FORMS")

using DataDrivenDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using DataDrivenSparse
using LinearAlgebra

# Generate library of candidate functions
# We have one variable u[1](t) = I(t), and three control parameters c[1]= beta, c[2]= zeta, c[3]= delta
@variables t u(t)[1:1] c[1:3]
u = collect(u)
c = collect(c)
# Define the library of candidate functions
h = Num[polynomial_basis(vcat(u,c),3); exp(c[1]*u[1]); exp(c[2]*u[1]); exp(c[3]*u[1]);
exp(c[1]*c[2]*u[1]); exp(c[1]*c[3]*u[1]); exp(c[2]*c[3]*u[1]);
exp(c[1]*c[2]*c[3]*u[1])]
# Define basis
basis = DataDrivenDiffEq.Basis(h, u, ctrls = c, iv = t)



