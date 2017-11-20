var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#LazySets.jl-1",
    "page": "Home",
    "title": "LazySets.jl",
    "category": "section",
    "text": "LazySets is a Julia package for calculus with convex sets.The aim is to provide a scalable library for solving complex set-based problems, such as those encountered in differential inclusions or reachability analysis techniques in the domain of formal verification. Typically, one is confronted with a set-based recurrence with a given initial set and/or input sets, and for visualization purposes the final result has to be obtained through an adequate projection onto low-dimensions. This library implements types to construct set formulas and methods to efficiently and accurately approximate the projection in low-dimensions.Pages = [\"index.md\"]"
},

{
    "location": "index.html#Introduction-1",
    "page": "Home",
    "title": "Introduction",
    "category": "section",
    "text": "The strategy consists of defining lazy (i.e. symbolic) representations that are used to write set-based formulas. This provides an exact but abstract formulation and may involve any common convex set class or operation between sets. Then, concrete information is obtained through querying specific directions. More precisely, each concrete subtype mathcalX of the abstract type LazySet, exports a method to calculate its support vector sigma(d mathcalX) in a given (arbitrary) direction d in mathbbR^n. Representing sets exactly but lazily has the advantage of being able to perform only the required operations on-demand.For very long computations (e.g. set-based recurrences with tens of thousands of elements), it is useful to combine both lazy and concrete representations such as polyhedral approximations. All this is easy to do with LazySets. Moreover, there is a specialized module for handling arrays of two-dimensional projections using Cartesian decomposition techniques. The projection can be taken to the desired precision using an iterative refinement method."
},

{
    "location": "index.html#Example-1",
    "page": "Home",
    "title": "Example",
    "category": "section",
    "text": "Let mathcalX_0 subset mathbbR^1000 be the Euclidean ball of center (1 ldots 1) and radius 01 in dimension n=1000. Suppose that given a real matrix A in mathbbR^1000 times 1000 we are interested in the equation:mathcalY = CH(e^A  mathcalX_0   BmathcalU mathcalX_0)where CH is the convex hull operator,  denotes Minkowski sum, mathcalU is a ball in the infinity norm centered at zero and radius 12, and B is a linear map of the appropriate dimensions.For concreteness, let's take A to be a random matrix with probability 1 of any entry being nonzero. Let's suppose that the input set mathcalU is two-dimensional, and that the linear map B is random. Using LazySets we can define this problem as follows:using LazySets\nA = sprandn(1000, 1000, 0.01)\nδ = 0.1\nX0 = Ball2(ones(1000), 0.1)\nB = randn(1000, 2)\nU = BallInf(zeros(2), 1.2)The @time macro reveals that building mathcalY with LazySets is instantaneous:@time Y = CH(SparseMatrixExp(A * δ) * X0 + δ * B * U, X0);\n0.000022 seconds (13 allocations: 16.094 KiB)By asking the concrete type of Y, we see that this object is a convex hull type, parameterized by the types of its arguments, corresponding to the mathematical formulation:julia> typeof(Y)\nLazySets.ConvexHull{LazySets.MinkowskiSum{LazySets.ExponentialMap{LazySets.Ball2},\nLazySets.LinearMap{LazySets.BallInf}},LazySets.Ball2}Now suppose that we are interested in observing the projection of mathcalY onto the variables number 1 and 500. First we define the 21000 projection matrix and apply it to mathcalY as a linear map (i.e. from the left). Second, we use the overapproximate method:proj_mat = [[1. zeros(1, 999)]; [zeros(1, 499) 1. zeros(1, 500)]]\n@time res = Approximations.overapproximate(proj_mat * Y);\n0.064034 seconds (1.12 k allocations: 7.691 MiB)We have calculated a box overapproximation of the exact projection onto the (x_1 x_500) plane. Notice that it takes about 0.064 seconds for the whole operation, allocating less than 10MB or RAM. Let us note that if the set operations were done explicitly, this would be much (!) slower. For instance, already the explicit computation of the matrix exponential would have costed 10x more, and allocated around 300MB. For even higher n, you'll probably run out of RAM! But this is doable with LazySets because the action of the matrix exponential over the set is being computed, evaluated only along the directions of interest. Similar comments apply to the Minkowski sums above.We can visualize the result using plot, as shown below (left-most plot).(Image: assets/example_ch.png)In the second and third plots, we have used a refined method that allows to specify a prescribed accuracy for the projection (in terms of Hausdorff distance). It can be passed as a second argument to overapproximate. Error tol. time (s) memory (MB)\n∞ (no refinement) 0.022 5.27\n1e-1 0.051 7.91\n1e-3 0.17 30.3This table shows the runtime and memory consumption for different error tolerances, and the results are shown in three plots of above, from left to right. When passing to a smaller tolerance, the corners connecting edges are more \"rounded\", at the expense of computational resources, since more support vectors have to be evaluated."
},

{
    "location": "index.html#Features-1",
    "page": "Home",
    "title": "Features",
    "category": "section",
    "text": "The core functionality of LazySets is:Lazy (i.e. symbolic) types for several classes of convex sets such as balls in different norms, polygons in constraint or vertex representation, special types such as lines and linear constraints, hyperrectangles, and high-dimensional polyhedra.\nMost commonly used set operations, e.g. Minkowski sum, Cartesian product, convex hull and interval hull approximations. Moreover, lazy linear maps and lazy exponential maps are also provided.On top of the previous basic type representations and operations, LazySets can be used to:Efficiently evaluate the support vector of nested lazy sets using parametrized LazySet arrays.\nCartesian decomposition of lazy sets using two-dimensional projections.\nFast overapproximation of an exact set using a polyhedral approximation, to the desired accuracy.\nExtensive visualization capabilities through Julia's Plots.jl framework."
},

{
    "location": "index.html#Manual-Outline-1",
    "page": "Home",
    "title": "Manual Outline",
    "category": "section",
    "text": "Pages = [\n    \"man/getting_started.md\",\n    \"man/polyhedral_approximations.md\",\n    \"man/decompose_example.md\",\n    \"man/fast_2d_LPs.md\"\n]\nDepth = 2"
},

{
    "location": "index.html#Library-Outline-1",
    "page": "Home",
    "title": "Library Outline",
    "category": "section",
    "text": "Pages = [\n    \"lib/representations.md\",\n    \"lib/operations.md\",\n    \"lib/approximations.md\",\n    \"lib/utils.md\"\n]\nDepth = 2"
},

{
    "location": "man/getting_started.html#",
    "page": "Getting Started",
    "title": "Getting Started",
    "category": "page",
    "text": ""
},

{
    "location": "man/getting_started.html#Getting-Started-1",
    "page": "Getting Started",
    "title": "Getting Started",
    "category": "section",
    "text": "In this section we review the recommended setup to start working with this package.Pages = [\"getting_started.md\"]"
},

{
    "location": "man/getting_started.html#Setup-1",
    "page": "Getting Started",
    "title": "Setup",
    "category": "section",
    "text": "This package requires Julia v0.6 or later. Refer to the official documentation on how to install it for your system. Below we explains the steps for setting up LazySets in your system and checking that it builds correctly."
},

{
    "location": "man/getting_started.html#Installation-1",
    "page": "Getting Started",
    "title": "Installation",
    "category": "section",
    "text": "To install LazySets and its dependencies, use the following commands inside Julia's REPL:Pkg.clone(\"https://github.com/acroy/Expokit.jl\")\nPkg.clone(\"https://github.com/JuliaReach/LazySets.jl\")The first command installs the dependency Expokit, that provides lazy matrix exponentiation routines."
},

{
    "location": "man/getting_started.html#Testing-1",
    "page": "Getting Started",
    "title": "Testing",
    "category": "section",
    "text": "Unit tests execute specific portions of the library code, checking that the result produced is the expected one. To run the unit tests run the following command in the terminal:$ julia --color=yes test/runtests.jlAlternatively, you can test the package in Julia's REPL with the command:julia> Pkg.test(\"LazySets\")"
},

{
    "location": "man/getting_started.html#Workflow-tips-1",
    "page": "Getting Started",
    "title": "Workflow tips",
    "category": "section",
    "text": "There are different ways to use Julia: from the terminal (so called REPL), from IJulia (i.e. Jupyter notebook), from Juno, ... If you don't have a preferred choice, we recommend using LazySets through IJulia; one reason is that the visualization is conveniently embedded into the notebook, and it can be exported into different formats, among other benefits. On the other hand, for development purposes you'll probably prefer using the REPL or the Juno environment."
},

{
    "location": "man/getting_started.html#Updating-1",
    "page": "Getting Started",
    "title": "Updating",
    "category": "section",
    "text": "After working with LazySets for some time, you may want to get the newest version. For this you can use this command:Pkg.checkout(\"LazySets\")That will checkout the latest version master branch, and precompile it the next time you enter a session and do using LazySets."
},

{
    "location": "man/polyhedral_approximations.html#",
    "page": "Polyhedral Approximations",
    "title": "Polyhedral Approximations",
    "category": "page",
    "text": ""
},

{
    "location": "man/polyhedral_approximations.html#Polyhedral-Approximations-1",
    "page": "Polyhedral Approximations",
    "title": "Polyhedral Approximations",
    "category": "section",
    "text": "In this section we review the mathematical notation and results from convex geometry that are used throughout LazySets."
},

{
    "location": "man/polyhedral_approximations.html#Preliminaries-1",
    "page": "Polyhedral Approximations",
    "title": "Preliminaries",
    "category": "section",
    "text": "Let us introduce some notation. Let mathbbI_n be the identity matrix of dimension ntimes n. For p geq 1, the p-norm of an n-dimensional vector x in mathbbR^n is denoted Vert x Vert_p."
},

{
    "location": "man/polyhedral_approximations.html#Support-Function-1",
    "page": "Polyhedral Approximations",
    "title": "Support Function",
    "category": "section",
    "text": "The support function is a basic notion for approximating convex sets. Let mathcalX subset mathbbR^n be a compact convex set. The support function of mathcalX is the function rho_mathcalX  mathbbR^nto mathbbR, defined asrho_mathcalX(ell) = maxlimits_x in mathcalX ell^mathrmT xWe recall the following elementary properties of the support function.Proposition. For all compact convex sets mathcalX, mathcalY in mathbbR^n, for all ntimes n real matrices M, all scalars lambda, and all vectors ell in mathbbR^n, we have:(1.1) rho_lambdamathcalX (ell) = rho_mathcalX (lambda ell), and         rho_lambdamathcalX (ell) = lambda rho_mathcalX (ell) if lambda  0.(1.2) rho_MmathcalX (ell) = rho_mathcalX (M^mathrmT ell)(1.3) rho_mathcalX oplus mathcalY (ell) = rho_mathcalX (ell) + rho_mathcalY (ell)(1.4) rho_mathcalX times mathcalY (ell) = ell^mathrmT sigma_mathcalX times mathcalY(ell)(1.5) rho_mathrmCH(mathcalXcupmathcalY) (ell) = max (rho_mathcalX (ell) rho_mathcalY (ell))"
},

{
    "location": "man/polyhedral_approximations.html#Support-Vector-1",
    "page": "Polyhedral Approximations",
    "title": "Support Vector",
    "category": "section",
    "text": "The farthest points of mathcalX in the direction ell  are the support vectors denoted sigma_mathcalX(ell). These points correspond to the optimal points for the support function, i.e.,sigma_mathcalX(ell) =  x in mathcalX  ell^mathrmT x  = rho_mathcalX(ell)  Since all support vectors in a given direction evaluate to the same value of the support function, we often speak of the support vector, where the choice of any support vector is implied.(Image: Illustration of the support function and the support vector)Proposition 2. Under the same conditions as in Proposition 1, the following hold:(2.1) sigma_lambdamathcalX (ell) = lambda sigma_mathcalX (lambda ell)(2.2) sigma_MmathcalX (ell) = Msigma_mathcalX (M^mathrmT ell)(2.3) sigma_mathcalX oplus mathcalY (ell) = sigma_mathcalX (ell) oplus sigma_mathcalY (ell)(2.4) sigma_mathcalX times mathcalY (ell) = (sigma_mathcalX(ell_1) sigma_mathcalY(ell_2)) ell = (ell_1 ell_2)(2.5) sigma_mathrmCH(mathcalXcupmathcalY) (ell) = textargmax_x y (ell^mathrmT x ell^mathrmT y),       where x in sigma_mathcalX(ell) y in sigma_mathcalY(ell)"
},

{
    "location": "man/polyhedral_approximations.html#Polyhedral-approximation-of-a-convex-set-1",
    "page": "Polyhedral Approximations",
    "title": "Polyhedral approximation of a convex set",
    "category": "section",
    "text": ""
},

{
    "location": "man/decompose_example.html#",
    "page": "Decomposing an Affine Map",
    "title": "Decomposing an Affine Map",
    "category": "page",
    "text": ""
},

{
    "location": "man/decompose_example.html#Decomposing-an-Affine-Map-1",
    "page": "Decomposing an Affine Map",
    "title": "Decomposing an Affine Map",
    "category": "section",
    "text": "In this section we present an illustrative example of the decomposed image of a linear map."
},

{
    "location": "man/fast_2d_LPs.html#",
    "page": "Fast 2D LPs",
    "title": "Fast 2D LPs",
    "category": "page",
    "text": ""
},

{
    "location": "man/fast_2d_LPs.html#Fast-2D-LPs-1",
    "page": "Fast 2D LPs",
    "title": "Fast 2D LPs",
    "category": "section",
    "text": "In this section we explain the implementation of the support vector for the case of convex polygons."
},

{
    "location": "man/fast_2d_LPs.html#Introduction-1",
    "page": "Fast 2D LPs",
    "title": "Introduction",
    "category": "section",
    "text": "Since vectors in the plane can be ordered by the angle with respect to the positive real axis, we can efficiently evaluate the support vector of a polygon in constraint representation by comparing normal directions, provided that its edges are ordered. We use the symbol preceq to compare directions, where the increasing direction is counter-clockwise.(Image: ../assets/intuition2dlp.png)"
},

{
    "location": "man/fast_2d_LPs.html#Algorithm-1",
    "page": "Fast 2D LPs",
    "title": "Algorithm",
    "category": "section",
    "text": ""
},

{
    "location": "lib/representations.html#",
    "page": "Common Set Representations",
    "title": "Common Set Representations",
    "category": "page",
    "text": ""
},

{
    "location": "lib/representations.html#Common-Set-Representations-1",
    "page": "Common Set Representations",
    "title": "Common Set Representations",
    "category": "section",
    "text": "This section of the manual describes the basic set representation types.Pages = [\"representations.md\"]\nDepth = 3CurrentModule = LazySets"
},

{
    "location": "lib/representations.html#LazySets",
    "page": "Common Set Representations",
    "title": "LazySets",
    "category": "Module",
    "text": "Main module for LazySets.jl – a Julia package for calculus with convex sets.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.LazySet",
    "page": "Common Set Representations",
    "title": "LazySets.LazySet",
    "category": "Type",
    "text": "LazySet\n\nAbstract type for a lazy set.\n\nEvery concrete LazySet must define a σ(d, X), representing the support vector of X in a given direction d, and dim, the ambient dimension of the set X.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.ρ",
    "page": "Common Set Representations",
    "title": "LazySets.ρ",
    "category": "Function",
    "text": "ρ(d::Vector{Float64}, sf::LazySet)::Float64\n\nEvaluate the support function of a set in a given direction.\n\nInput\n\nd  – a real vector, the direction investigated\nsf – a convex set\n\nOutput\n\nρ(d, sf) – the support function\n\n\n\n"
},

{
    "location": "lib/representations.html#Abstract-support-function-and-support-vector-1",
    "page": "Common Set Representations",
    "title": "Abstract support function and support vector",
    "category": "section",
    "text": "LazySets\nLazySets.LazySet\nρ"
},

{
    "location": "lib/representations.html#Balls-1",
    "page": "Common Set Representations",
    "title": "Balls",
    "category": "section",
    "text": "Unit balls are defined by int center (vector) and radius (scalar), such as infinity-norm balls,B_infty(c r) =  x  mathbbR^n  Vert x - cVert_infty leq r and Euclidean (2-norm) balls,B_2(c r) =  x  mathbbR^n  Vert x - cVert_2 leq r "
},

{
    "location": "lib/representations.html#LazySets.Ball2",
    "page": "Common Set Representations",
    "title": "LazySets.Ball2",
    "category": "Type",
    "text": "Ball2 <: LazySet\n\nType that represents a ball in the 2-norm.\n\nFields\n\ncenter – center of the ball as a real vector\nradius – radius of the ball as a scalar ( 0)\n\nExamples\n\nA five-dimensional ball in the 2-norm centered at the origin of radius 0.5:\n\njulia> using LazySets\njulia> B = Ball2(zeros(5), 0.5)\nLazySets.Ball2([0.0, 0.0, 0.0, 0.0, 0.0], 0.5)\njulia> dim(B)\n5\n\nWe evaluate the support vector in a given direction:\n\njulia> σ(ones(5), B)\n5-element Array{Float64,1}:\n0.06742\n0.13484\n0.20226\n0.26968\n0.3371\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.dim-Tuple{LazySets.Ball2}",
    "page": "Common Set Representations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(B)\n\nReturn the dimension of a Ball2.\n\nInput\n\nB – a ball in the 2-norm\n\nOutput\n\nThe ambient dimension of the ball.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.Ball2}",
    "page": "Common Set Representations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d, B)\n\nReturn the support vector of a Ball2 in a given direction.\n\nInput\n\nd – a direction\nB – a ball in the 2-norm\n\nOutput\n\nThe support vector in the given direction.\n\nNotes\n\nIf the given direction has norm zero, the origin is returned.\n\n\n\n"
},

{
    "location": "lib/representations.html#Euclidean-norm-ball-1",
    "page": "Common Set Representations",
    "title": "Euclidean norm ball",
    "category": "section",
    "text": "Ball2\ndim(B::Ball2)\nσ(d::AbstractVector{Float64}, B::Ball2)"
},

{
    "location": "lib/representations.html#LazySets.BallInf",
    "page": "Common Set Representations",
    "title": "LazySets.BallInf",
    "category": "Type",
    "text": "BallInf <: LazySet\n\nType that represents a ball in the infinity norm.\n\nFields\n\ncenter – center of the ball as a real vector\nradius – radius of the ball as a scalar ( 0)\n\nExamples\n\nWe create the two-dimensional unit ball, and compute its support function along the direction (1 1):\n\njulia> B = BallInf(zeros(2), 0.1)\nLazySets.BallInf([0.0, 0.0], 0.1)\n\njulia> dim(B)\n2\n\njulia> ρ([1., 1.], B)\n0.2\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.dim-Tuple{LazySets.BallInf}",
    "page": "Common Set Representations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(B)\n\nReturn the dimension of a BallInf.\n\nInput\n\nB – a ball in the infinity norm\n\nOutput\n\nThe ambient dimension of the ball.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.BallInf}",
    "page": "Common Set Representations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d, B)\n\nReturn the support vector of an infinity-norm ball in a given direction.\n\nInput\n\nd – direction\nB – unit ball in the infinity norm\n\nAlgorithm\n\nThis code is a vectorized version of\n\n[(d[i] >= 0) ? B.center[i] + B.radius : B.center[i] - B.radius for i in 1:length(d)]\n\nNotice that we cannot use B.center + sign.(d) * B.radius, since the built-in sign function is such that sign(0) = 0, instead of 1. For this reason, we use the custom unit_step function, that allows to do: B.center + unit_step.(d) * B.radius (the dot operator performs broadcasting, to accept vector-valued entries).\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.vertices_list-Tuple{LazySets.BallInf}",
    "page": "Common Set Representations",
    "title": "LazySets.vertices_list",
    "category": "Method",
    "text": "vertices_list(B::BallInf)\n\nReturn the list of vertices of a ball in the infinity norm.\n\nInput\n\nB – a ball in the infinity norm\n\nOutput\n\nThe list of vertices as an array of floating-point vectors.\n\nNotes\n\nFor high-dimensions, it is preferable to develop a vertex_iterator approach.\n\n\n\n"
},

{
    "location": "lib/representations.html#Base.LinAlg.norm",
    "page": "Common Set Representations",
    "title": "Base.LinAlg.norm",
    "category": "Function",
    "text": "norm(B::BallInf, [p])\n\nReturn the norm of a BallInf. It is the norm of the enclosing ball (of the given norm) of minimal volume.\n\nInput\n\nB – ball in the infinity norm\np – (optional, default: Inf) norm\n\nOutput\n\nA real number representing the norm.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.radius",
    "page": "Common Set Representations",
    "title": "LazySets.radius",
    "category": "Function",
    "text": "radius(B::BallInf, [p])\n\nReturn the radius of a ball in the infinity norm. It is the radius of the enclosing ball (of the given norm) of minimal volume with the same center.\n\nInput\n\nB – a ball in the infinity norm\np – (optional, default: Inf) norm\n\nOutput\n\nA real number representing the radius.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.diameter",
    "page": "Common Set Representations",
    "title": "LazySets.diameter",
    "category": "Function",
    "text": "diameter(B::BallInf, [p])\n\nReturn the diameter of a ball in the infinity norm. It is the maximum distance between any two elements of the set, or, equivalently, the diameter of the enclosing ball (of the given norm) of minimal volume with the same center.\n\nInput\n\nB – a ball in the infinity norm\np – (optional, default: Inf) norm\n\nOutput\n\nA real number representing the diameter.\n\n\n\n"
},

{
    "location": "lib/representations.html#Infinity-norm-ball-1",
    "page": "Common Set Representations",
    "title": "Infinity norm ball",
    "category": "section",
    "text": "BallInf\ndim(B::BallInf)\nσ(d::AbstractVector{Float64}, B::BallInf)\nvertices_list(B::BallInf)\nnorm(B::BallInf, p::Real=Inf)\nradius(B::BallInf, p::Real=Inf)\ndiameter(B::BallInf, p::Real=Inf)"
},

{
    "location": "lib/representations.html#Polygons-1",
    "page": "Common Set Representations",
    "title": "Polygons",
    "category": "section",
    "text": ""
},

{
    "location": "lib/representations.html#LazySets.HPolygon",
    "page": "Common Set Representations",
    "title": "LazySets.HPolygon",
    "category": "Type",
    "text": "HPolygon <: LazySet\n\nType that represents a convex polygon in constraint representation, whose edges are sorted in counter-clockwise fashion with respect to their normal directions.\n\nFields\n\nconstraints_list –  an array of linear constraints\n\nNote\n\nThe HPolygon constructor does not perform sorting of the given list of edges. Use addconstraint! to iteratively add and sort the edges.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.addconstraint!-Tuple{LazySets.HPolygon,LazySets.LinearConstraint}",
    "page": "Common Set Representations",
    "title": "LazySets.addconstraint!",
    "category": "Method",
    "text": "addconstraint!(P, constraint)\n\nAdd a linear constraint to a polygon in constraint representation keeping the constraints sorted by their normal directions.\n\nInput\n\nP          – a polygon\nconstraint – the linear constraint to add, see LinearConstraint\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.dim-Tuple{LazySets.HPolygon}",
    "page": "Common Set Representations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(P)\n\nReturn the ambient dimension of the polygon.\n\nInput\n\nP – polygon in constraint representation\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.HPolygon}",
    "page": "Common Set Representations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d, P)\n\nReturn the support vector of a polygon in a given direction. Return the zero vector if there are no constraints (i.e., the universal polytope).\n\nInput\n\nd – direction\nP – polygon in constraint representation\n\nAlgorithm\n\nComparison of directions is performed using polar angles, see the overload of <= for two-dimensional vectors.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.is_contained-Tuple{AbstractArray{Float64,1},LazySets.HPolygon}",
    "page": "Common Set Representations",
    "title": "LazySets.is_contained",
    "category": "Method",
    "text": "is_contained(x, P)\n\nReturn whether a given vector is contained in the polygon.\n\nInput\n\nx – two-dimensional vector\nP – polygon in constraint representation\n\nOutput\n\nReturn true iff x ∈ P.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.tovrep-Tuple{LazySets.HPolygon}",
    "page": "Common Set Representations",
    "title": "LazySets.tovrep",
    "category": "Method",
    "text": "tovrep(P)\n\nBuild a vertex representation of the given polygon.\n\nInput\n\nP – polygon in constraint representation\n\nOutput\n\nThe same polygon but in vertex representation, VPolygon.\n\nNote\n\nThe linear constraints of the input HPolygon are assumed to be sorted by their normal directions in counter-clockwise fashion.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.vertices_list-Tuple{LazySets.HPolygon}",
    "page": "Common Set Representations",
    "title": "LazySets.vertices_list",
    "category": "Method",
    "text": "vertices_list(P)\n\nReturn the list of vertices of a convex polygon in constraint representation.\n\nInput\n\nP – polygon in constraint representation\n\nOutput\n\nList of vertices as an array of vertex pairs, Vector{Vector{Float64}}.\n\n\n\n"
},

{
    "location": "lib/representations.html#Constraint-representation-1",
    "page": "Common Set Representations",
    "title": "Constraint representation",
    "category": "section",
    "text": "HPolygon\naddconstraint!(P::HPolygon, c::LinearConstraint)\ndim(P::HPolygon)\nσ(d::AbstractVector{Float64}, P::HPolygon)\n\nis_contained(x::AbstractVector{Float64}, P::HPolygon)\ntovrep(P::HPolygon)\nvertices_list(P::HPolygon)"
},

{
    "location": "lib/representations.html#LazySets.HPolygonOpt",
    "page": "Common Set Representations",
    "title": "LazySets.HPolygonOpt",
    "category": "Type",
    "text": "HPolygonOpt <: LazySet\n\nType that represents a convex polygon in constraint representation, whose edges are sorted in counter-clockwise fashion with respect to their normal directions. This is a refined version of HPolygon.\n\nFields\n\nP   – polygon\nind – an index in the list of constraints to begin the search to          evaluate the support functions.\n\nNotes\n\nThis structure is optimized to evaluate the support function/vector with a large sequence of directions, which are one to one close. The strategy is to have an index that can be used to warm-start the search for optimal values in the support vector computation.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.addconstraint!-Tuple{LazySets.HPolygonOpt,LazySets.LinearConstraint}",
    "page": "Common Set Representations",
    "title": "LazySets.addconstraint!",
    "category": "Method",
    "text": "addconstraint!(P, constraint)\n\nAdd a linear constraint to an optimized polygon in constraint representation, keeping the constraints sorted by their normal directions.\n\nInput\n\nP          – optimized polygon\nconstraint – the linear constraint to add, see LinearConstraint\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.dim-Tuple{LazySets.HPolygonOpt}",
    "page": "Common Set Representations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(P)\n\nReturn the ambient dimension of the optimized polygon.\n\nInput\n\nP – optimized polygon\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.HPolygonOpt}",
    "page": "Common Set Representations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d, P)\n\nReturn the support vector of the optimized polygon in a given direction.\n\nInput\n\nd – direction\nP – optimized polygon in constraint representation\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.is_contained-Tuple{AbstractArray{Float64,1},LazySets.HPolygonOpt}",
    "page": "Common Set Representations",
    "title": "LazySets.is_contained",
    "category": "Method",
    "text": "is_contained(x, P)\n\nReturn whether a given vector is contained in an optimized polygon in constraint representation.\n\nInput\n\nx – two-dimensional vector\nP – optimized polygon in constraint representation\n\nOutput\n\nReturn true iff x ∈ P.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.tovrep-Tuple{LazySets.HPolygonOpt}",
    "page": "Common Set Representations",
    "title": "LazySets.tovrep",
    "category": "Method",
    "text": "tovrep(P)\n\nBuild a vertex representation of the given optimized polygon.\n\nInput\n\nP – optimized polygon in constraint representation\n\nOutput\n\nThe same polygon in vertex representation, VPolygon.\n\nNote\n\nThe linear constraints of the input HPolygonOpt are assumed to be sorted by their normal directions in counter-clockwise fashion.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.vertices_list-Tuple{LazySets.HPolygonOpt}",
    "page": "Common Set Representations",
    "title": "LazySets.vertices_list",
    "category": "Method",
    "text": "vertices_list(P)\n\nReturn the list of vertices of a convex polygon in constraint representation.\n\nInput\n\nP – an optimized polygon in constraint representation\n\nOutput\n\nList of vertices as an array of vertex pairs, Vector{Vector{Float64}}.\n\n\n\n"
},

{
    "location": "lib/representations.html#Optimized-constraint-representation-1",
    "page": "Common Set Representations",
    "title": "Optimized constraint representation",
    "category": "section",
    "text": "HPolygonOpt\naddconstraint!(P::HPolygonOpt, c::LinearConstraint)\ndim(P::HPolygonOpt)\nσ(d::AbstractVector{Float64}, P::HPolygonOpt)\n\nis_contained(x::AbstractVector{Float64}, P::HPolygonOpt)\ntovrep(P::HPolygonOpt)\nvertices_list(P::HPolygonOpt)"
},

{
    "location": "lib/representations.html#LazySets.VPolygon",
    "page": "Common Set Representations",
    "title": "LazySets.VPolygon",
    "category": "Type",
    "text": "VPolygon <: LazySet\n\nType that represents a polygon by its vertices.\n\nFields\n\nvertices_list – the list of vertices\n\nNotes\n\nThe constructor of VPolygon runs a convex hull algorithm, and the given vertices are sorted in counter-clockwise fashion. If you don't want to take the convex hull, set the apply_convex_hull=false flag when instantiating the constructor.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.dim-Tuple{LazySets.VPolygon}",
    "page": "Common Set Representations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(P)\n\nReturn the ambient dimension of the polygon.\n\nInput\n\nP – polygon in vertex representation\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.vertices_list-Tuple{LazySets.VPolygon}",
    "page": "Common Set Representations",
    "title": "LazySets.vertices_list",
    "category": "Method",
    "text": "vertices_list(P)\n\nReturn the list of vertices of a convex polygon in vertex representation.\n\nInput\n\nP – a polygon given in vertex representation\n\nOutput\n\nList of vertices as an array of vertex pairs, Vector{Vector{Float64}}.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.singleton_list-Tuple{LazySets.VPolygon}",
    "page": "Common Set Representations",
    "title": "LazySets.singleton_list",
    "category": "Method",
    "text": "singleton_list(P)\n\nReturn the vertices of a convex polygon in vertex representation as a list of singletons.\n\nInput\n\nP – a polygon given in vertex representation\n\nOutput\n\nList of vertices as an array of vertex pairs, Vector{Singleton{Float64}}.\n\n\n\n"
},

{
    "location": "lib/representations.html#Vertex-representation-1",
    "page": "Common Set Representations",
    "title": "Vertex representation",
    "category": "section",
    "text": "VPolygon\ndim(P::VPolygon)\nvertices_list(P::VPolygon)\nsingleton_list(P::VPolygon)"
},

{
    "location": "lib/representations.html#LazySets.intersection",
    "page": "Common Set Representations",
    "title": "LazySets.intersection",
    "category": "Function",
    "text": "intersection(Δ1, Δ2)\n\nReturn the intersection of two 2D lines.\n\nInput\n\nΔ1 – a line\nΔ2 – another line\n\nOutput\n\nThe intersection point.\n\nExamples\n\nThe line y = -x + 1 intersected with y = x:\n\njulia> intersection(Line([1., 1.], 1.), Line([-1., 1.], 0.))\n2-element Array{Float64,1}:\n 0.5\n 0.5\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.LinearConstraint",
    "page": "Common Set Representations",
    "title": "LazySets.LinearConstraint",
    "category": "Type",
    "text": "LinearConstraint\n\nType that represents a linear constraint (a half-space) of the form a⋅x ≦ b.\n\nFields\n\na –  a normal direction\nb – the constraint\n\nEXAMPLES:\n\nThe set y >= 0 in the plane:\n\njulia> LinearConstraint([0, -1.], 0.)\nLazySets.LinearConstraint([0.0, -1.0], 0.0)\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.Line",
    "page": "Common Set Representations",
    "title": "LazySets.Line",
    "category": "Type",
    "text": "Line\n\nType that represents a line in 2D of the form a⋅x = b.\n\nFields\n\na  – a normal direction (size = 2)\nb  – the constraint\n\nExamples\n\nThe line y = -x + 1:\n\njulia> Line([1., 1.], 1.)\nLazySets.Line([1.0, 1.0], 1.0)\n\n\n\n"
},

{
    "location": "lib/representations.html#Lines-and-linear-constraints-1",
    "page": "Common Set Representations",
    "title": "Lines and linear constraints",
    "category": "section",
    "text": "intersection\nLinearConstraint\nLine"
},

{
    "location": "lib/representations.html#LazySets.Hyperrectangle",
    "page": "Common Set Representations",
    "title": "LazySets.Hyperrectangle",
    "category": "Type",
    "text": "Hyperrectangle <: LazySet\n\nType that represents a hyperrectangle.\n\nA hyperrectangle is the Cartesian product of one-dimensional intervals.\n\nFields\n\ncenter – center of the hyperrectangle as a real vector\nradius – radius of the ball as a real vector, i.e., half of its width along             each coordinate direction\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.dim-Tuple{LazySets.Hyperrectangle}",
    "page": "Common Set Representations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(H)\n\nReturn the dimension of a Hyperrectangle.\n\nInput\n\nH – a hyperrectangle\n\nOutput\n\nThe ambient dimension of the hyperrectangle as an integer.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.Hyperrectangle}",
    "page": "Common Set Representations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d, H)\n\nReturn the support vector of a Hyperrectangle in a given direction.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.vertices_list-Tuple{LazySets.Hyperrectangle}",
    "page": "Common Set Representations",
    "title": "LazySets.vertices_list",
    "category": "Method",
    "text": "vertices_list(H::Hyperrectangle)\n\nReturn the vertices of a hyperrectangle.\n\nInput\n\nH – a hyperrectangle\n\nOutput\n\nThe list of vertices as an array of floating-point vectors.\n\nNotes\n\nFor high-dimensions, it is preferable to develop a vertex_iterator approach.\n\n\n\n"
},

{
    "location": "lib/representations.html#Base.LinAlg.norm",
    "page": "Common Set Representations",
    "title": "Base.LinAlg.norm",
    "category": "Function",
    "text": "norm(H::Hyperrectangle, [p])\n\nReturn the norm of a Hyperrectangle. It is the norm of the enclosing ball (of the given norm) of minimal volume.\n\nInput\n\nH – hyperrectangle\np – (optional, default: Inf) norm\n\nOutput\n\nA real number representing the norm.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.radius",
    "page": "Common Set Representations",
    "title": "LazySets.radius",
    "category": "Function",
    "text": "radius(H::Hyperrectangle, [p])\n\nReturn the radius of a hyperrectangle. It is the radius of the enclosing ball (of the given norm) of minimal volume with the same center.\n\nInput\n\nH – hyperrectangle\np – (optional, default: Inf) norm\n\nOutput\n\nA real number representing the radius.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.diameter",
    "page": "Common Set Representations",
    "title": "LazySets.diameter",
    "category": "Function",
    "text": "diameter(H::Hyperrectangle, [p])\n\nReturn the diameter of a hyperrectangle. It is the maximum distance between any two elements of the set, or, equivalently, the diameter of the enclosing ball (of the given norm) of minimal volume with the same center.\n\nInput\n\nH – a hyperrectangle\np – (optional, default: Inf) norm\n\nOutput\n\nA real number representing the diameter.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.high-Tuple{LazySets.Hyperrectangle}",
    "page": "Common Set Representations",
    "title": "LazySets.high",
    "category": "Method",
    "text": "high(H::Hyperrectangle)\n\nReturn the higher coordinates of a hyperrectangle.\n\nInput\n\nH – a hyperrectangle\n\nOutput\n\nA vector with the higher coordinates of the hyperrectangle, one entry per dimension.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.low-Tuple{LazySets.Hyperrectangle}",
    "page": "Common Set Representations",
    "title": "LazySets.low",
    "category": "Method",
    "text": "low(H::Hyperrectangle)\n\nReturn the lower coordinates of a hyperrectangle.\n\nInput\n\nH – a hyperrectangle\n\nOutput\n\nA vector with the lower coordinates of the hyperrectangle, one entry per dimension.\n\n\n\n"
},

{
    "location": "lib/representations.html#Hyperrectangles-1",
    "page": "Common Set Representations",
    "title": "Hyperrectangles",
    "category": "section",
    "text": "Hyperrectangle\ndim(H::Hyperrectangle)\nσ(d::AbstractVector{Float64}, H::Hyperrectangle)\nvertices_list(H::Hyperrectangle)\nnorm(H::Hyperrectangle, p::Real=Inf)\nradius(H::Hyperrectangle, p::Real=Inf)\ndiameter(H::Hyperrectangle, p::Real=Inf)\nhigh(H::Hyperrectangle)\nlow(H::Hyperrectangle)"
},

{
    "location": "lib/representations.html#LazySets.VoidSet",
    "page": "Common Set Representations",
    "title": "LazySets.VoidSet",
    "category": "Type",
    "text": "VoidSet <: LazySet\n\nType that represents a void (neutral) set with respect to Minkowski sum.\n\nFields\n\ndim – ambient dimension of the VoidSet\n\n\n\n"
},

{
    "location": "lib/representations.html#VoidSets-1",
    "page": "Common Set Representations",
    "title": "VoidSets",
    "category": "section",
    "text": "VoidSet"
},

{
    "location": "lib/representations.html#LazySets.Singleton",
    "page": "Common Set Representations",
    "title": "LazySets.Singleton",
    "category": "Type",
    "text": "Singleton <: LazySet\n\nType that represents a singleton, that is, a set with a unique element.\n\nFields\n\nelement – the only element of the set\n\n\n\n"
},

{
    "location": "lib/representations.html#Singletons-1",
    "page": "Common Set Representations",
    "title": "Singletons",
    "category": "section",
    "text": "Singleton"
},

{
    "location": "lib/operations.html#",
    "page": "Common Set Operations",
    "title": "Common Set Operations",
    "category": "page",
    "text": ""
},

{
    "location": "lib/operations.html#Common-Set-Operations-1",
    "page": "Common Set Operations",
    "title": "Common Set Operations",
    "category": "section",
    "text": "This section of the manual describes the basic symbolic types describing operations between sets.Pages = [\"operations.md\"]\nDepth = 3CurrentModule = LazySets"
},

{
    "location": "lib/operations.html#LazySets.MinkowskiSum",
    "page": "Common Set Operations",
    "title": "LazySets.MinkowskiSum",
    "category": "Type",
    "text": "MinkowskiSum <: LazySet\n\nType that represents the Minkowski sum of two convex sets.\n\nFields\n\nX – a convex set\nY – a convex set\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.dim-Tuple{LazySets.MinkowskiSum}",
    "page": "Common Set Operations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(ms)\n\nAmbient dimension of a Minkowski product.\n\nInput\n\nms – Minkowski sum\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.MinkowskiSum}",
    "page": "Common Set Operations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d::AbstractVector{Float64}, ms::MinkowskiSum)\n\nSupport vector of a Minkowski sum.\n\nInput\n\nd  – vector\nms – Minkowski sum\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.MinkowskiSumArray",
    "page": "Common Set Operations",
    "title": "LazySets.MinkowskiSumArray",
    "category": "Type",
    "text": "MinkowskiSumArray <: LazySet\n\nType that represents the Minkowski sum of a finite number of sets.\n\nFields\n\nsfarray – array of sets\n\nNotes\n\nThis type is optimized to be used on the left-hand side of additions only.\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.dim-Tuple{LazySets.MinkowskiSumArray}",
    "page": "Common Set Operations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(ms::MinkowskiSumArray)\n\nAmbient dimension of the Minkowski sum of a finite number of sets.\n\nInput\n\nms – Minkowski sum array\n\nNotes\n\nWe do not double-check that the dimensions always match.\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.MinkowskiSumArray}",
    "page": "Common Set Operations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d::Vector{Float64}, ms::MinkowskiSumArray)\n\nSupport vector of the Minkowski sum of a finite number of sets.\n\nInput\n\nd – direction\nms – Minkowski sum array\n\n\n\n"
},

{
    "location": "lib/operations.html#Base.:+-Tuple{LazySets.MinkowskiSumArray,LazySets.LazySet}",
    "page": "Common Set Operations",
    "title": "Base.:+",
    "category": "Method",
    "text": "+(msa, sf)\n\nAdds the support function to the array.\n\nInput\n\nmsa – Minkowski sum array\nsf – general support function\n\nNotes\n\nThis function is overridden for more specific types of sf.\n\n\n\n"
},

{
    "location": "lib/operations.html#Base.:+-Tuple{LazySets.MinkowskiSumArray,LazySets.MinkowskiSumArray}",
    "page": "Common Set Operations",
    "title": "Base.:+",
    "category": "Method",
    "text": "+(msa1, msa2)\n\nAppends the elements of the second array to the first array.\n\nInput\n\nmsa1 – first Minkowski sum array\nmsa2 – second Minkowski sum array\n\n\n\n"
},

{
    "location": "lib/operations.html#Base.:+-Tuple{LazySets.MinkowskiSumArray,LazySets.VoidSet}",
    "page": "Common Set Operations",
    "title": "Base.:+",
    "category": "Method",
    "text": "+(msa, vs)\n\nReturns the original array because addition with a void set is a no-op.\n\nInput\n\nmsa – Minkowski sum array\nvs – void set\n\n\n\n"
},

{
    "location": "lib/operations.html#Minkowski-Sum-1",
    "page": "Common Set Operations",
    "title": "Minkowski Sum",
    "category": "section",
    "text": "MinkowskiSum\ndim(ms::MinkowskiSum)\nσ(d::AbstractVector{Float64}, ms::MinkowskiSum)MinkowskiSumArray\ndim(ms::MinkowskiSumArray)\nσ(d::AbstractVector{Float64}, ms::MinkowskiSumArray)\nBase.:+(msa::MinkowskiSumArray, sf::LazySet)\nBase.:+(msa1::MinkowskiSumArray, msa2::MinkowskiSumArray)\nBase.:+(msa::MinkowskiSumArray, sf::VoidSet)"
},

{
    "location": "lib/operations.html#LazySets.ConvexHull",
    "page": "Common Set Operations",
    "title": "LazySets.ConvexHull",
    "category": "Type",
    "text": "ConvexHull <: LazySet\n\nType that represents the convex hull of the union of two convex sets.\n\nFields\n\nX – a convex set\nY – another convex set\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.dim-Tuple{LazySets.ConvexHull}",
    "page": "Common Set Operations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(P)\n\nReturn the ambient dimension of the convex hull of two sets.\n\nInput\n\nch – the convex hull of two sets\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.ConvexHull}",
    "page": "Common Set Operations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d, P)\n\nReturn the support vector of a convex hull in a given direction.\n\nInput\n\nd  – direction\nch – the convex hull of two sets\n\n\n\n"
},

{
    "location": "lib/operations.html#Convex-Hull-1",
    "page": "Common Set Operations",
    "title": "Convex Hull",
    "category": "section",
    "text": "ConvexHull\ndim(ch::ConvexHull)\nσ(d::AbstractVector{Float64}, ch::ConvexHull)"
},

{
    "location": "lib/operations.html#LazySets.convex_hull",
    "page": "Common Set Operations",
    "title": "LazySets.convex_hull",
    "category": "Function",
    "text": "convex_hull(points; algorithm)\n\nCompute the convex hull of points in the plane.\n\nInput\n\npoints    – array of vectors containing the 2D coordinates of the points\nalgorithm – (optional, default: \"monotone_chain\") choose the convex                hull algorithm, valid options are:\n\"monotone_chain\"\n\nOutput\n\nThe convex hull as a list of 2D vectors with the coordinates of the points.\n\nExamples\n\nCompute the convex hull of a random set of points:\n\njulia> points = [randn(2) for i in 1:30]; # 30 random points in 2D\njulia> hull = convex_hull(points);\njulia> typeof(hull)\nArray{Array{Float64,1},1}\n\nWe can plot the random points, and the polygon whose vertices are the computed convex hull, using Plots:\n\njulia> using Plots\njulia> plot([Tuple(pi) for pi in points], seriestype=:scatter)\njulia> plot!(VPolygon(hull), alpha=0.2)\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.convex_hull!",
    "page": "Common Set Operations",
    "title": "LazySets.convex_hull!",
    "category": "Function",
    "text": "convex_hull!(points; algorithm)\n\nCompute the convex hull of points in the plane, in-place. See also: convex_hull.\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.right_turn",
    "page": "Common Set Operations",
    "title": "LazySets.right_turn",
    "category": "Function",
    "text": "right_turn(O, A, B)\n\nDetermine if the acute angle defined by the three points O, A, B in the plane is a right turn (counter-clockwise) with respect to the center O.\n\nInput\n\nO – center point\nA – one point\nB – another point\n\nAlgorithm\n\nThe cross product is used to determine the sense of rotation. If the result is 0, the points are collinear; if it is positive, the three points constitute a positive angle of rotation around O from A to B; otherwise a negative angle.\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.monotone_chain!",
    "page": "Common Set Operations",
    "title": "LazySets.monotone_chain!",
    "category": "Function",
    "text": "monotone_chain!(points)\n\nCompute the convex hull of points in the plane using Andrew's monotone chain method.\n\nInput\n\npoints – array of vectors containing the 2D coordinates of the points;             is sorted in-place inside this function\n\nOutput\n\nArray of vectors containing the 2D coordinates of the corner points of the convex hull.\n\nNotes\n\nFor large sets of points, it is convenient to use static vectors to get maximum performance. For information on how to convert usual vectors into static vectors, see the type SVector provided by the StaticArrays package.\n\nAlgorithm\n\nThis function implements Andrew's monotone chain convex hull algorithm to construct the convex hull of a set of n points in the plane in O(n log n) time. For further details see the wikipedia page: Monotone chain\n\n\n\n"
},

{
    "location": "lib/operations.html#Convex-Hull-Algorithms-1",
    "page": "Common Set Operations",
    "title": "Convex Hull Algorithms",
    "category": "section",
    "text": "convex_hull\nconvex_hull!\nright_turn\nmonotone_chain!"
},

{
    "location": "lib/operations.html#LazySets.CartesianProduct",
    "page": "Common Set Operations",
    "title": "LazySets.CartesianProduct",
    "category": "Type",
    "text": "CartesianProduct <: LazySet\n\nType that represents the cartesian product.\n\nFields\n\nX – convex set\nY – another convex set\n\nFor the cartesian product a several sets, there exists a special type CartesianProductArray. \n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.dim-Tuple{LazySets.CartesianProduct}",
    "page": "Common Set Operations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(cp)\n\nAmbient dimension of a Cartesian product.\n\nInput\n\ncp – cartesian product\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.CartesianProduct}",
    "page": "Common Set Operations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d, cp)\n\nSupport vector of a Cartesian product.\n\nInput\n\nd – direction\ncp – cartesian product\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.is_contained-Tuple{AbstractArray{Float64,1},LazySets.CartesianProduct}",
    "page": "Common Set Operations",
    "title": "LazySets.is_contained",
    "category": "Method",
    "text": "is_contained(d, cp)\n\nReturn whether a vector belongs to a given cartesian product set.\n\nInput\n\nd    –  a vector\ncp   – a cartesian product\n\nOutput\n\nReturn true iff d ∈ cp.\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.CartesianProductArray",
    "page": "Common Set Operations",
    "title": "LazySets.CartesianProductArray",
    "category": "Type",
    "text": "CartesianProductArray <: LazySet\n\nType that represents the cartesian product of a finite number of sets.\n\nFields\n\nsfarray – array of sets\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.dim-Tuple{LazySets.CartesianProductArray}",
    "page": "Common Set Operations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(cp)\n\nAmbient dimension of the Cartesian product of a finite number of sets.\n\nInput\n\ncp – cartesian product array\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.CartesianProductArray}",
    "page": "Common Set Operations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d, cp)\n\nSupport vector of the Cartesian product of a finite number of sets.\n\nInput\n\nd – direction\ncp – cartesian product array\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.is_contained-Tuple{AbstractArray{Float64,1},LazySets.CartesianProductArray}",
    "page": "Common Set Operations",
    "title": "LazySets.is_contained",
    "category": "Method",
    "text": "is_contained(d, cp)\n\nReturn whether a given vector is contained in the cartesian product of a finite number of sets.\n\nInput\n\nd – vector\ncp – cartesian product array\n\n\n\n"
},

{
    "location": "lib/operations.html#Cartesian-Product-1",
    "page": "Common Set Operations",
    "title": "Cartesian Product",
    "category": "section",
    "text": "CartesianProduct\ndim(cp::CartesianProduct)\nσ(d::AbstractVector{Float64}, cp::CartesianProduct)\nis_contained(d::AbstractVector{Float64}, cp::CartesianProduct)CartesianProductArray\ndim(cp::CartesianProductArray)\nσ(d::AbstractVector{Float64}, cp::CartesianProductArray)\nis_contained(d::AbstractVector{Float64}, cp::CartesianProductArray)"
},

{
    "location": "lib/operations.html#LazySets.LinearMap",
    "page": "Common Set Operations",
    "title": "LazySets.LinearMap",
    "category": "Type",
    "text": "LinearMap <: LazySet\n\nType that represents a linear transform of a set. This class is a wrapper around a linear transformation MS of a set S, such that it changes the behaviour of the support vector of the new set.\n\nFields\n\nM  – a linear map, which can a be densem matrix, sparse matrix or a subarray object\nsf – a convex set represented by its support function\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.dim-Tuple{LazySets.LinearMap}",
    "page": "Common Set Operations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(lm)\n\nAmbient dimension of the linear map of a set.\n\nIt corresponds to the output dimension of the linear map.\n\nInput\n\nlm – a linear map\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.LinearMap}",
    "page": "Common Set Operations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d, lm)\n\nSupport vector of the linear map of a set.\n\nIf S = MB, where M is sa matrix and B is a set, it follows that σ(d, S) = Mσ(M^T d, B) for any direction d.\n\nInput\n\nd  – a direction\nlm – a linear map\n\n\n\n"
},

{
    "location": "lib/operations.html#Linear-Maps-1",
    "page": "Common Set Operations",
    "title": "Linear Maps",
    "category": "section",
    "text": "LinearMap\ndim(lm::LinearMap)\nσ(d::AbstractVector{Float64}, lm::LinearMap)"
},

{
    "location": "lib/operations.html#LazySets.ExponentialMap",
    "page": "Common Set Operations",
    "title": "LazySets.ExponentialMap",
    "category": "Type",
    "text": "ExponentialMap <: LazySet\n\nType that represents the action of an exponential map on a set.\n\nFields\n\nspmexp  – a matrix exponential\nX      – a convex set represented by its support function\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.dim-Tuple{LazySets.ExponentialMap}",
    "page": "Common Set Operations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(em)\n\nThe ambient dimension of a ExponentialMap.\n\nInput\n\nem – an ExponentialMap\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.ExponentialProjectionMap}",
    "page": "Common Set Operations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d, eprojmap)\n\nSupport vector of an ExponentialProjectionMap.\n\nInput\n\nd         – a direction\neprojmap  – the projection of an exponential map\n\nIf S = (LMR)B, where L and R are dense matrices, M is a matrix exponential, and B is a set, it follows that: σ(d, S) = LMR σ(R^T M^T L^T d, B) for any direction d.\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.ExponentialProjectionMap",
    "page": "Common Set Operations",
    "title": "LazySets.ExponentialProjectionMap",
    "category": "Type",
    "text": "ExponentialProjectionMap\n\nType that represents the application of the projection of a SparseMatrixExp over a given set.\n\nFields\n\nspmexp   – the projection of an exponential map\nX       – a set represented by its support function\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.dim-Tuple{LazySets.ExponentialProjectionMap}",
    "page": "Common Set Operations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(eprojmap)\n\nThe ambient dimension of a ExponentialProjectionMap.\n\nIt is given by the output dimension (left-most matrix).\n\nInput\n\neprojmap – an ExponentialProjectionMap\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.ProjectionSparseMatrixExp",
    "page": "Common Set Operations",
    "title": "LazySets.ProjectionSparseMatrixExp",
    "category": "Type",
    "text": "ProjectionSparseMatrixExp\n\nType that represents the projection of a SparseMatrixExp.\n\nFields\n\nL – left multiplication matrix\nE – the exponential of a sparse matrix\nR – right multiplication matrix\n\nOutput\n\nA type that abstract the matrix operation L * exp(E.M) * R, for a given sparse matrix E.M.\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.SparseMatrixExp",
    "page": "Common Set Operations",
    "title": "LazySets.SparseMatrixExp",
    "category": "Type",
    "text": "SparseMatrixExp\n\nType that represents the matrix exponential of a sparse matrix, and provides evaluation of its action on vectors.\n\nFields\n\nM – sparse matrix\n\nNotes\n\nThis class is provided for use with very large and very sparse matrices. The evaluation of the exponential matrix action over vectores relies on the Expokit package. \n\n\n\n"
},

{
    "location": "lib/operations.html#Exponential-Maps-1",
    "page": "Common Set Operations",
    "title": "Exponential Maps",
    "category": "section",
    "text": "ExponentialMap\ndim(emap::ExponentialMap)\nσ(d::AbstractVector{Float64}, eprojmap::ExponentialProjectionMap)ExponentialProjectionMap\ndim(eprojmap::ExponentialProjectionMap)ProjectionSparseMatrixExp\nSparseMatrixExp"
},

{
    "location": "lib/approximations.html#",
    "page": "Approximations",
    "title": "Approximations",
    "category": "page",
    "text": ""
},

{
    "location": "lib/approximations.html#Approximations-1",
    "page": "Approximations",
    "title": "Approximations",
    "category": "section",
    "text": "This section of the manual describes the Cartesian decomposition algorithms and the approximation of high-dimensional convex sets using projections.Pages = [\"approximations.md\"]\nDepth = 3CurrentModule = LazySets.Approximations"
},

{
    "location": "lib/approximations.html#LazySets.Approximations.decompose",
    "page": "Approximations",
    "title": "LazySets.Approximations.decompose",
    "category": "Function",
    "text": "decompose(X)\n\nCompute an overapproximation of the projections of the given set over each two-dimensional subspace using box directions.\n\nInput\n\nX  – lazy set\n\nOutput\n\nA CartesianProductArray corresponding to the cartesian product of 2x2 polygons.\n\n\n\ndecompose(X, ɛi)\n\nCompute an overapproximation of the projections of the given set over each two-dimensional subspace with a certified error bound.\n\nInput\n\nX  – lazy set\nɛi – array with the error bound for each projection (different error bounds         can be passed to different blocks)\n\nOutput\n\nA CartesianProductArray corresponding to the cartesian product of 2x2 polygons.\n\nAlgorithm\n\nThis algorithm assumes a decomposition into two-dimensional subspaces only, i.e. partitions of the form 2 2  2. In particular if X is a CartesianProductArray, no check is performed to verify that assumption.\n\nIt proceeds as follows:\n\nProject the set X into each partition, with MX, where M is the identity matrix in the block coordinates and zero otherwise.\nOverapproximate the set with a given error bound, ɛi[i], for i = 1  b,\nReturn the result as an array of support functions.\n\n\n\ndecompose(X, ɛ)\n\nCompute an overapproximation of the projections of the given set over each two-dimensional subspace with a certified error bound.\n\nThis function is a particular case of decompose(X, ɛi), where the same error bound for each block is assumed.\n\nInput\n\nX  – lazy set\nɛ –  error bound\n\nOutput\n\nA CartesianProductArray corresponding to the cartesian product of 2x2 polygons.\n\n\n\n"
},

{
    "location": "lib/approximations.html#LazySets.Approximations.overapproximate",
    "page": "Approximations",
    "title": "LazySets.Approximations.overapproximate",
    "category": "Function",
    "text": "overapproximate(X)\n\nReturn an approximation of the given 2D set as a box-shaped polygon.\n\nInput\n\nX – lazy set, assumed to be two-dimensional\n\nOutput\n\nA polygon in constraint representation.\n\n\n\noverapproximate(X, ɛ)\n\nReturn an ɛ-close approximation of the given 2D set (in terms of Hausdorff distance) as a polygon.\n\nInput\n\nX – lazy set, assumed to be two-dimensional\nɛ – the error bound\n\nOutput\n\nA polygon in constraint representation.\n\n\n\n"
},

{
    "location": "lib/approximations.html#Cartesian-Decomposition-1",
    "page": "Approximations",
    "title": "Cartesian Decomposition",
    "category": "section",
    "text": "decompose\noverapproximate"
},

{
    "location": "lib/approximations.html#LazySets.Approximations.ballinf_approximation",
    "page": "Approximations",
    "title": "LazySets.Approximations.ballinf_approximation",
    "category": "Function",
    "text": "ballinf_approximation(X)\n\nOverapproximation of a set by a ball in the infinity norm.\n\nInput\n\nX – a lazy set\n\nOutput\n\nH – a ball in the infinity norm which tightly contains the given set\n\nAlgorithm\n\nThe center and radius of the box are obtained by evaluating the support function of the given set along the canonical directions.\n\n\n\n"
},

{
    "location": "lib/approximations.html#LazySets.Approximations.box_approximation",
    "page": "Approximations",
    "title": "LazySets.Approximations.box_approximation",
    "category": "Function",
    "text": "box_approximation(X)\n\nOverapproximate a set by a box (hyperrectangle).\n\nInput\n\nX – a lazy set\n\nOutput\n\nH – a (tight) hyperrectangle\n\nAlgorithm\n\nThe center of the hyperrectangle is obtained by averaring the support function of the given set in the canonical directions, and the lengths of the sides can be recovered from the distance among support functions in the same directions.\n\n\n\n"
},

{
    "location": "lib/approximations.html#LazySets.Approximations.box_approximation_symmetric",
    "page": "Approximations",
    "title": "LazySets.Approximations.box_approximation_symmetric",
    "category": "Function",
    "text": "box_approximation_symmetric(X)\n\nOverapproximation of a set by a hyperrectangle which contains the origin.\n\nInput\n\nX – a lazy set\n\nOuptut\n\nH – a symmetric interval around the origin which tightly contains the given set\n\nAlgorithm\n\nThe center of the box is the origin, and the radius is obtained by computing the maximum value of the support function evaluated at the canonical directions.\n\n\n\n"
},

{
    "location": "lib/approximations.html#LazySets.Approximations.box_approximation_helper",
    "page": "Approximations",
    "title": "LazySets.Approximations.box_approximation_helper",
    "category": "Function",
    "text": "box_approximation_helper(X)\n\nCommon code of box_approximation and box_approximation_symmetric.\n\nInput\n\nX – a lazy set\n\nOutput\n\nH – a (tight) hyperrectangle\n\nAlgorithm\n\nThe center of the hyperrectangle is obtained by averaring the support function the given set in the canonical directions, and the lengths of the sides can be recovered from the distance among support functions in the same directions.\n\n\n\n"
},

{
    "location": "lib/approximations.html#Box-Approximations-1",
    "page": "Approximations",
    "title": "Box Approximations",
    "category": "section",
    "text": "ballinf_approximation\nbox_approximation\nbox_approximation_symmetric\nbox_approximation_helper"
},

{
    "location": "lib/approximations.html#Base.LinAlg.norm",
    "page": "Approximations",
    "title": "Base.LinAlg.norm",
    "category": "Function",
    "text": "norm(X::LazySet, [p])\n\nReturn the norm of a LazySet. It is the norm of the enclosing ball (of the given norm) of minimal volume.\n\nInput\n\nX – a lazy set\np – (optional, default: Inf) norm\n\nOutput\n\nA real number representing the norm.\n\n\n\n"
},

{
    "location": "lib/approximations.html#LazySets.radius",
    "page": "Approximations",
    "title": "LazySets.radius",
    "category": "Function",
    "text": "radius(X::LazySet, [p])\n\nReturn the radius of a LazySet. It is the radius of the enclosing ball (of the given norm) of minimal volume with the same center.\n\nInput\n\nX – lazy set\np – (optional, default: Inf) norm\n\nOutput\n\nA real number representing the radius.\n\n\n\n"
},

{
    "location": "lib/approximations.html#LazySets.diameter",
    "page": "Approximations",
    "title": "LazySets.diameter",
    "category": "Function",
    "text": "diameter(X::LazySet, [p])\n\nReturn the diameter of a LazySet. It is the maximum distance between any two elements of the set, or, equivalently, the diameter of the enclosing ball (of the given norm) of minimal volume with the same center.\n\nInput\n\nX – lazy set\np – (optional, default: Inf) norm\n\nOutput\n\nA real number representing the diameter.\n\n\n\n"
},

{
    "location": "lib/approximations.html#Metric-properties-of-sets-1",
    "page": "Approximations",
    "title": "Metric properties of sets",
    "category": "section",
    "text": "norm(X::LazySet, p::Real=Inf)\nradius(X::LazySet, p::Real=Inf)\ndiameter(X::LazySet, p::Real=Inf)"
},

{
    "location": "lib/approximations.html#LazySets.Approximations.approximate",
    "page": "Approximations",
    "title": "LazySets.Approximations.approximate",
    "category": "Function",
    "text": "approximate(X, ɛ)\n\nReturn an ɛ-close approximation of the given 2D set (in terms of Hausdorff distance) as an inner and an outer approximation composed by sorted local Approximation2D.\n\nInput\n\nX – a 2D set defined by its support function\nɛ – the error bound\n\n\n\n"
},

{
    "location": "lib/approximations.html#Iterative-refinement-1",
    "page": "Approximations",
    "title": "Iterative refinement",
    "category": "section",
    "text": "approximate"
},

{
    "location": "lib/utils.html#",
    "page": "Utility Functions",
    "title": "Utility Functions",
    "category": "page",
    "text": ""
},

{
    "location": "lib/utils.html#LazySets.unit_step-Tuple{Float64}",
    "page": "Utility Functions",
    "title": "LazySets.unit_step",
    "category": "Method",
    "text": "unit_step(x)\n\nThe unit step function, which returns 1 if and only if x is greater or equal than zero.\n\nInput\n\nx – a floating point number\n\nNotes\n\nThis function can be used with vector-valued arguments via the dot operator.\n\nExamples\n\njulia> unit_step([-0.6, 1.3, 0.0])\n3-element Array{Float64,1}:\n -1.0\n 1.0\n 1.0\n\n\n\n"
},

{
    "location": "lib/utils.html#LazySets.jump2pi",
    "page": "Utility Functions",
    "title": "LazySets.jump2pi",
    "category": "Function",
    "text": "jump2pi(x)\n\nReturn x + 2 and only if x is negative.\n\nInput\n\nx – a floating point number\n\nExamples\n\njulia> jump2pi(0.0)\n0.0\njulia> jump2pi(-0.5)\n5.783185307179586\njulia> jump2pi(0.5)\n0.5\n\n\n\n"
},

{
    "location": "lib/utils.html#Base.:<=-Tuple{AbstractArray{Float64,1},AbstractArray{Float64,1}}",
    "page": "Utility Functions",
    "title": "Base.:<=",
    "category": "Method",
    "text": "u <= v\n\nStates if arg(u) [2π] <= arg(v) [2π].\n\nInput\n\nu –  a first direction\nv –  a second direction\n\nOutput\n\nTrue iff arg(u) [2π] <= arg(v) [2π]\n\nNotes\n\nThe argument is measured in counter-clockwise fashion, with the 0 being the direction (1, 0).\n\n\n\n"
},

{
    "location": "lib/utils.html#Utility-functions-1",
    "page": "Utility Functions",
    "title": "Utility functions",
    "category": "section",
    "text": "unit_step(x::Float64)\njump2pi\nBase.:<=(u::AbstractVector{Float64}, v::AbstractVector{Float64})"
},

{
    "location": "about.html#",
    "page": "About",
    "title": "About",
    "category": "page",
    "text": ""
},

{
    "location": "about.html#About-1",
    "page": "About",
    "title": "About",
    "category": "section",
    "text": "This page contains some general information about this project, and recommendations about contributing.Pages = [\"about.md\"]"
},

{
    "location": "about.html#Contributing-1",
    "page": "About",
    "title": "Contributing",
    "category": "section",
    "text": "If you like this package, consider contributing! You can send bug reports (or fix them and send your code), add examples to the documentation or propose new features.Below some conventions that we follow when contributing to this package are detailed. For specific guidelines on documentation, see the Documentations Guidelines wiki."
},

{
    "location": "about.html#Branches-1",
    "page": "About",
    "title": "Branches",
    "category": "section",
    "text": "Each pull request (PR) should be pushed in a new branch with the name of the author followed by a descriptive name, e.g. t/mforets/my_feature. If the branch is associated to a previous discussion in one issue, we use the name of the issue for easier lookup, e.g. t/mforets/7."
},

{
    "location": "about.html#Unit-testing-and-continuous-integration-(CI)-1",
    "page": "About",
    "title": "Unit testing and continuous integration (CI)",
    "category": "section",
    "text": "This project is synchronized with Travis CI, such that each PR gets tested before merging (and the build is automatically triggered after each new commit). For the maintainability of this project, it is important to understand and fix the failing doctests if they exist. We develop in Julia v0.6.0, but for experimentation we also build on the nightly branch.To run the unit tests locally, you should do:$ julia --color=yes test/runtests.jl"
},

{
    "location": "about.html#Contributing-to-the-documentation-1",
    "page": "About",
    "title": "Contributing to the documentation",
    "category": "section",
    "text": "This documentation is written in Markdown, and it relies on Documenter.jl to produce the HTML layout. To build the docs, run make.jl:$ julia --color=yes docs/make.jl"
},

{
    "location": "about.html#Related-Projects-1",
    "page": "About",
    "title": "Related Projects",
    "category": "section",
    "text": "The project 3PLIB is a Java Library developed by Frédéric Viry, and it is one of the previous works that led to the creation of LazySets.jl. 3PLIB is specialized to planar projections of convex polyhedra. It was initially created to embed this feature in Java applications, and also provides a backend for visualization of high-dimensional reach set approximations computed with SpaceEx."
},

{
    "location": "about.html#Credits-1",
    "page": "About",
    "title": "Credits",
    "category": "section",
    "text": "These persons have contributed to LazySets.jl (in alphabetic order):Marcelo Forets\nChristian Schilling\nFrédéric ViryWe are also grateful to Goran Frehse for enlightening discussions."
},

]}
