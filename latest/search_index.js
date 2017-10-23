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
    "text": "LazySets is a Julia package for calculus with convex sets."
},

{
    "location": "index.html#Features-1",
    "page": "Home",
    "title": "Features",
    "category": "section",
    "text": "At the core of LazySets there are:Lazy (i.e. symbolic) types for several convex sets such as convex polygons, different classes of polytopes, and special types such as linear constraints.\nMost commonly used set operations, e.g. Minkowski sum, Cartesian product, convex hull and interval hull approximations. Moreover, lazy linear maps and lazy exponential maps are also provided.Each instance of the abstract type LazySet implements a function, sigma(d mathcalX), to compute the supoprt vector of a set mathcalXsubset mathbbR^n in a given (arbitrary) direction d in mathbbR^n. This has the advantage of being able to perform only the required operations on-demand.On top of the previous basic type representations and operations, the following functionality is available:Efficient evaluation of the support vector of nested lazy sets.\nCartesian decomposition of lazy sets using support vectors.\nFast overapproximation of symbolic set computations using a polyhedral approximation."
},

{
    "location": "index.html#Manual-Outline-1",
    "page": "Home",
    "title": "Manual Outline",
    "category": "section",
    "text": "Pages = [\n    \"man/getting_started.md\",\n    \"man/support_vectors.md\",\n    \"man/polyhedral_approximations.md\",\n    \"man/fast_2d_LPs.md\"\n]\nDepth = 2"
},

{
    "location": "index.html#Library-Outline-1",
    "page": "Home",
    "title": "Library Outline",
    "category": "section",
    "text": "Pages = [\n    \"lib/representations.md\",\n    \"lib/operations.md\",\n    \"lib/approximations.md\"\n]\nDepth = 2"
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
    "text": "In this section we set the mathematical notation and review the theoretical background from convex geometry that is used throughout LazySets. Then, we present an illustrative example of the decomposed image of a linear map.Pages = [\"getting_started.md\"]"
},

{
    "location": "man/getting_started.html#Preliminaries-1",
    "page": "Getting Started",
    "title": "Preliminaries",
    "category": "section",
    "text": "Let us introduce some notation. Let mathbbI_n be the identity matrix of dimension ntimes n. For p geq 1, the p-norm of an n-dimensional vector x in mathbbR^n is denoted  Vert x Vert_p."
},

{
    "location": "man/getting_started.html#Support-Function-1",
    "page": "Getting Started",
    "title": "Support Function",
    "category": "section",
    "text": "The support function is a basic notion for approximating convex sets. Let mathcalX subset mathbbR^n be a compact convex set. The support function of mathcalX is the function rho_mathcalX  mathbbR^nto mathbbR, defined asrho_mathcalX(ell) = maxlimits_x in mathcalX ell^mathrmT xWe recall the following elementary properties of the support function.Proposition. For all compact convex sets mathcalX, mathcalY in mathbbR^n, for all ntimes n real matrices M, all scalars lambda, and all vectors ell in mathbbR^n, we have:(1.1) rho_lambdamathcalX (ell) = rho_mathcalX (lambda ell), and         rho_lambdamathcalX (ell) = lambda rho_mathcalX (ell) if lambda  0.(1.2) rho_MmathcalX (ell) = rho_mathcalX (M^mathrmT ell)(1.3) rho_mathcalX oplus mathcalY (ell) = rho_mathcalX (ell) + rho_mathcalY (ell)(1.4) rho_mathcalX times mathcalY (ell) = ell^mathrmT sigma_mathcalX times mathcalY(ell)(1.5) rho_mathrmCH(mathcalXcupmathcalY) (ell) = max (rho_mathcalX (ell) rho_mathcalY (ell))"
},

{
    "location": "man/getting_started.html#Support-Vector-1",
    "page": "Getting Started",
    "title": "Support Vector",
    "category": "section",
    "text": "The farthest points of mathcalX in the direction ell  are the support vectors denoted sigma_mathcalX(ell). These points correspond to the optimal points for the support function, i.e.,sigma_mathcalX(ell) =  x in mathcalX  ell^mathrmT x  = rho_mathcalX(ell)  Since all support vectors in a given direction evaluate to the same value of the support function, we often speak of the support vector, where the choice of any support vector is implied.(Image: Illustration of the support function and the support vector)Proposition 2. Under the same conditions as in Proposition 1, the following hold:(2.1) sigma_lambdamathcalX (ell) = lambda sigma_mathcalX (lambda ell)(2.2) sigma_MmathcalX (ell) = Msigma_mathcalX (M^mathrmT ell)(2.3) sigma_mathcalX oplus mathcalY (ell) = sigma_mathcalX (ell) oplus sigma_mathcalY (ell)(2.4) sigma_mathcalX times mathcalY (ell) = (sigma_mathcalX(ell_1) sigma_mathcalY(ell_2)) ell = (ell_1 ell_2)(2.5) sigma_mathrmCH(mathcalXcupmathcalY) (ell) = textargmax_x y (ell^mathrmT x ell^mathrmT y),       where x in sigma_mathcalX(ell) y in sigma_mathcalY(ell)"
},

{
    "location": "man/getting_started.html#Polyhedral-Approximation-of-a-Convex-Set-1",
    "page": "Getting Started",
    "title": "Polyhedral Approximation of a Convex Set",
    "category": "section",
    "text": ""
},

{
    "location": "man/getting_started.html#Example:-decomposing-an-affine-map-1",
    "page": "Getting Started",
    "title": "Example: decomposing an affine map",
    "category": "section",
    "text": ""
},

{
    "location": "man/support_vectors.html#",
    "page": "Support Vectors",
    "title": "Support Vectors",
    "category": "page",
    "text": ""
},

{
    "location": "man/polyhedral_approximations.html#",
    "page": "Polyhedral Approximations",
    "title": "Polyhedral Approximations",
    "category": "page",
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
    "text": "This section of the manual describes the basic set representation types.Pages = [\"representations.md\"]CurrentModule = LazySets"
},

{
    "location": "lib/representations.html#LazySets.Hyperrectangle",
    "page": "Common Set Representations",
    "title": "LazySets.Hyperrectangle",
    "category": "Type",
    "text": "Hyperrectangle <: LazySet\n\nType that represents a Hyperrectangle.\n\nA hyperrectangle is the Cartesian product of one-dimensional intervals.\n\nFields\n\ncenter – center of the hyperrectangle as a real vector\nradius – radius of the ball as a real vector, i.e. its width along             each coordinate direction\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.BallInf",
    "page": "Common Set Representations",
    "title": "LazySets.BallInf",
    "category": "Type",
    "text": "BallInf <: LazySet\n\nType that represents a ball in the infinity norm.\n\nFields\n\ncenter – center of the ball as a real vector\nradius – radius of the ball as a scalar ( 0)\n\nExamples\n\nWe create the two-dimensional unit ball, and compute its support function along the direction (1 1):\n\njulia> B = BallInf(zeros(2), 0.1)\nLazySets.BallInf([0.0, 0.0], 0.1)\n\njulia> dim(B)\n2\n\njulia> ρ([1., 1.], B)\n0.2\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.Ball2",
    "page": "Common Set Representations",
    "title": "LazySets.Ball2",
    "category": "Type",
    "text": "Ball2 <: LazySet\n\nType that represents a ball in the 2-norm.\n\nFields\n\ncenter – center of the ball as a real vector\nradius – radius of the ball as a scalar ( 0)\n\nExamples\n\nA five-dimensional ball in the 2-norm centered at the origin of radius 0.5:\n\njulia> using LazySets\njulia> B = Ball2(zeros(5), 0.5)\nLazySets.Ball2([0.0, 0.0, 0.0, 0.0, 0.0], 0.5)\njulia> dim(B)\n5\n\nWe evaluate the support vector in a given direction:\n\njulia> σ(ones(5), B)\n5-element Array{Float64,1}:\n0.06742\n0.13484\n0.20226\n0.26968\n0.3371\n\n\n\n"
},

{
    "location": "lib/representations.html#Balls-1",
    "page": "Common Set Representations",
    "title": "Balls",
    "category": "section",
    "text": "Unit balls are defined by int center (vector) and radius (scalar), such as infinity-norm balls,B_infty(c r) =  x  mathbbR^n  Vert x - cVert_infty leq r and Euclidean (2-norm) balls,B_2(c r) =  x  mathbbR^n  Vert x - cVert_2 leq r Hyperrectangle\nBallInf\nBall2"
},

{
    "location": "lib/representations.html#LazySets.HPolygon",
    "page": "Common Set Representations",
    "title": "LazySets.HPolygon",
    "category": "Type",
    "text": "HPolygon <: LazySet\n\nType that represents a convex polygon (in H-representation).\n\nFields\n\nconstraints –  an array of linear constraints\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.VPolygon",
    "page": "Common Set Representations",
    "title": "LazySets.VPolygon",
    "category": "Type",
    "text": "VPolygon\n\nType that represents a polygon by its vertices.\n\nFields\n\nvl – the list of vertices\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.plot_polygon",
    "page": "Common Set Representations",
    "title": "LazySets.plot_polygon",
    "category": "Function",
    "text": "plot_polygon(P, backend, [name], [gridlines])\n\nPlot a polygon given in constraint form.\n\nInput\n\nP – a polygon, given as a HPolygon or the refined class HPolygonOpt\nbackend – (optional, default: pyplot): select the plot backend; valid              options are:\npyplot_savefig – use PyPlot package, save to a file\npyplot_inline  – use PyPlot package, showing in external program\ngadfly         – use Gadfly package, showing in browser\n''             – (empty string), return nothing, without plotting\nname – (optional, default: plot.png) the filename of the plot, if it is           saved to disk\ngridlines – (optional, default: false) to display or not gridlines in                the output plot\n\nExamples\n\nThis function can receive one polygon, as in:\n\njulia> using LazySets, PyPlot\njulia> H = HPolygon([LinearConstraint([1.0, 0.0], 0.6), LinearConstraint([0.0, 1.0], 0.6),\n       LinearConstraint([-1.0, 0.0], -0.4), LinearConstraint([0.0, -1.0], -0.4)])\njulia> plot_polygon(H, backend=\"pyplot_inline\");\n\nMultiple polygons can be plotted passing a list instead of a single element:\n\njulia> Haux = HPolygon([LinearConstraint([1.0, 0.0], 1.2), LinearConstraint([0.0, 1.0], 1.2),\n       LinearConstraint([-1.0, 0.0], -0.8), LinearConstraint([0.0, -1.0], -0.8)])\njulia> plot_polygon([H, Haux], backend=\"pyplot_inline\");\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.tovrep",
    "page": "Common Set Representations",
    "title": "LazySets.tovrep",
    "category": "Function",
    "text": "tovrep(s)\n\nBuild a vertex representation of the given polygon.\n\nInput\n\ns – a polygon in H-representation, HPolygon. The linear constraints are        assumed sorted by their normal directions.\n\nOutput\n\nThe same polygon in a vertex representation, VPolygon.\n\n\n\ntovrep(po)\n\nBuild a vertex representation of the given polygon.\n\nInput\n\npo – a polygon in H-representation. The linear constraints are         assumed sorted by their normal directions.\n\nOutput\n\nThe same polygon in a vertex representation.\n\n\n\n"
},

{
    "location": "lib/representations.html#Polygons-1",
    "page": "Common Set Representations",
    "title": "Polygons",
    "category": "section",
    "text": "HPolygon\nVPolygon\nplot_polygon\ntovrep"
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
    "location": "lib/representations.html#LazySets.VoidSet",
    "page": "Common Set Representations",
    "title": "LazySets.VoidSet",
    "category": "Type",
    "text": "VoidSet <: LazySet\n\nType that represents a void (neutral) set with respect to Minkowski sum.\n\nFields\n\ndim – ambient dimension of the VoidSet \n\n\n\n"
},

{
    "location": "lib/representations.html#VoidSet-1",
    "page": "Common Set Representations",
    "title": "VoidSet",
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
    "location": "lib/representations.html#Singleton-1",
    "page": "Common Set Representations",
    "title": "Singleton",
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
    "text": "This section of the manual describes the basic symbolic types describing operations between sets.Pages = [\"operations.md\"]CurrentModule = LazySets"
},

{
    "location": "lib/operations.html#LazySets.MinkowskiSum",
    "page": "Common Set Operations",
    "title": "LazySets.MinkowskiSum",
    "category": "Type",
    "text": "MinkowskiSum <: LazySet\n\nType that represents the Minkowski sum of two convex sets.\n\nFields\n\nX – a convex set\nY – a convex set\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.MinkowskiSumArray",
    "page": "Common Set Operations",
    "title": "LazySets.MinkowskiSumArray",
    "category": "Type",
    "text": "MinkowskiSumArray <: LazySet\n\nType that represents the Minkowski sum of a finite number of sets.\n\nFields\n\nsfarray – array of sets\n\nNotes\n\nThis type is optimized to be used on the left-hand side of additions only.\n\n\n\n"
},

{
    "location": "lib/operations.html#Minkowski-Sum-1",
    "page": "Common Set Operations",
    "title": "Minkowski Sum",
    "category": "section",
    "text": "MinkowskiSum\nMinkowskiSumArray"
},

{
    "location": "lib/operations.html#LazySets.ConvexHull",
    "page": "Common Set Operations",
    "title": "LazySets.ConvexHull",
    "category": "Type",
    "text": "ConvexHull <: LazySet\n\nType that represents the convex hull of the union of two convex sets.\n\nFields\n\nX – a convex set\nY – another convex set\n\n\n\n"
},

{
    "location": "lib/operations.html#Convex-Hull-1",
    "page": "Common Set Operations",
    "title": "Convex Hull",
    "category": "section",
    "text": "ConvexHull"
},

{
    "location": "lib/operations.html#LazySets.CartesianProduct",
    "page": "Common Set Operations",
    "title": "LazySets.CartesianProduct",
    "category": "Type",
    "text": "CartesianProduct <: LazySet\n\nType that represents the cartesian product.\n\nFields\n\nX – convex set\nY – another convex set\n\nFor the cartesian product a several sets, there exists a special type CartesianProductArray. \n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.CartesianProductArray",
    "page": "Common Set Operations",
    "title": "LazySets.CartesianProductArray",
    "category": "Type",
    "text": "CartesianProductArray <: LazySet\n\nType that represents the cartesian product of a finite number of sets.\n\nFields\n\nsfarray – array of sets\n\n\n\n"
},

{
    "location": "lib/operations.html#Cartesian-Product-1",
    "page": "Common Set Operations",
    "title": "Cartesian Product",
    "category": "section",
    "text": "CartesianProduct\nCartesianProductArray"
},

{
    "location": "lib/operations.html#LazySets.LinearMap",
    "page": "Common Set Operations",
    "title": "LazySets.LinearMap",
    "category": "Type",
    "text": "LinearMap <: LazySet\n\nType that represents a linear transform of a set. This class is a wrapper around a linear transformation MS of a set S, such that it changes the behaviour of the support vector of the new set.\n\nFields\n\nM  – a linear map, which can a be densem matrix, sparse matrix or a subarray object\nsf – a convex set represented by its support function\n\n\n\n"
},

{
    "location": "lib/operations.html#Linear-Maps-1",
    "page": "Common Set Operations",
    "title": "Linear Maps",
    "category": "section",
    "text": "LinearMap"
},

{
    "location": "lib/operations.html#LazySets.ExponentialMap",
    "page": "Common Set Operations",
    "title": "LazySets.ExponentialMap",
    "category": "Type",
    "text": "ExponentialMap <: LazySet\n\nType that represents the action of an exponential map on a set.\n\nFields\n\nspmexp  – a matrix exponential\nX      – a convex set represented by its support function\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.ExponentialProjectionMap",
    "page": "Common Set Operations",
    "title": "LazySets.ExponentialProjectionMap",
    "category": "Type",
    "text": "ExponentialProjectionMap\n\nType that represents the application of the projection of a SparseMatrixExp over a given set.\n\nFields\n\nspmexp   – the projection of an exponential map\nX       – a set represented by its support function\n\n\n\n"
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
    "location": "lib/operations.html#LazySets.σ-Tuple{Union{Array{Float64,1}, SparseVector{Float64,Int64}},LazySets.ExponentialProjectionMap}",
    "page": "Common Set Operations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d, eprojmap)\n\nSupport vector of an ExponentialProjectionMap.\n\nInput\n\nd         – a direction\neprojmap  – the projection of an exponential map\n\nIf S = (LMR)B, where L and R are dense matrices, M is a matrix exponential, and B is a set, it follows that: σ(d, S) = LMR σ(R^T M^T L^T d, B) for any direction d.\n\n\n\n"
},

{
    "location": "lib/operations.html#Exponential-Maps-1",
    "page": "Common Set Operations",
    "title": "Exponential Maps",
    "category": "section",
    "text": "ExponentialMap\nExponentialProjectionMap\nProjectionSparseMatrixExp\nSparseMatrixExp\nσ(d::Union{Vector{Float64}, SparseVector{Float64,Int64}},\n           eprojmap::ExponentialProjectionMap)"
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
    "text": "This section of the manual describes the Cartesian decomposition algorithms and the approximation of high-dimensional convex sets using projections.Pages = [\"approximations.md\"]CurrentModule = LazySets.Approximations"
},

{
    "location": "lib/approximations.html#LazySets.Approximations.decompose",
    "page": "Approximations",
    "title": "LazySets.Approximations.decompose",
    "category": "Function",
    "text": "decompose(X)\n\nCompute an overapproximation of the projections of the given set over each two-dimensional subspace using box directions.\n\nInput\n\nX  – set represented by support functions\n\nOutput\n\nA CartesianProductArray corresponding to the cartesian product of 2x2 polygons.\n\n\n\ndecompose(X, ɛi)\n\nCompute an overapproximation of the projections of the given set over each two-dimensional subspace with a certified error bound.\n\nInput\n\nX  – set represented by support functions\nɛi – array, error bound for each projection (different error bounds         can be passed to different blocks)\n\nOutput\n\nA CartesianProductArray corresponding to the cartesian product of 2x2 polygons.\n\nAlgorithm\n\nThis algorithm assumes a decomposition into two-dimensional subspaces only, i.e. partitions of the form 2 2  2. In particular if X is a CartesianProductArray no check is performed to verify that assumption.\n\nIt proceeds as follows:\n\nProject the set X into each partition, with MX, where M is the identity matrix in the block coordinates and zero otherwise.\nOverapproximate the set with a given error bound, ɛi[i], for i = 1  b,\nReturn the result as an array of support functions.\n\n\n\ndecompose(X, ɛ)\n\nCompute an overapproximation of the projections of the given set over each two-dimensional subspace with a certified error bound.\n\nThis function is a particular case of decompose(X, ɛi), where the same error bound for each block is assumed. \n\nInput\n\nX  – set represented by support functions\nɛ –  error bound\n\nOutput\n\nA CartesianProductArray corresponding to the cartesian product of 2x2 polygons.\n\n\n\n"
},

{
    "location": "lib/approximations.html#LazySets.Approximations.overapproximate",
    "page": "Approximations",
    "title": "LazySets.Approximations.overapproximate",
    "category": "Function",
    "text": "overapproximate(S)\n\nReturn an approximation of the given 2D set as a polygon, using box directions.\n\nInput\n\nS – a 2D set defined by its support function\n\nOutput\n\nA polygon in constraint representation.\n\n\n\noverapproximate(S, ɛ)\n\nReturn an ɛ-close approximation of the given 2D set (in terms of Hausdorff distance) as a polygon.\n\nInput\n\nS – a 2D set defined by its support function\nɛ – the error bound\n\nOutput\n\nA polygon in constraint representation.\n\n\n\n"
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
    "text": "ballinf_approximation(S)\n\nOverapproximation of a set by a ball in the infinity norm.\n\nInput\n\nX – a lazy set\n\nOutput\n\nH – a ball in the infinity norm which tightly contains the given set\n\nAlgorithm\n\nThe center and radius of the box are obtained by evaluating the support function of the given set along the canonical directions.\n\n\n\n"
},

{
    "location": "lib/approximations.html#LazySets.Approximations.box_approximation",
    "page": "Approximations",
    "title": "LazySets.Approximations.box_approximation",
    "category": "Function",
    "text": "box_approximation(X)\n\nOverapproximate a set by a box (hyperrectangle). \n\nInput\n\nX – a lazy set\n\nOutput\n\nH – a (tight) hyperrectangle\n\nAlgorithm\n\nThe center of the hyperrectangle is obtained by averaring the support function of the given set in the canonical directions, and the lengths of the sides can be recovered from the distance among support functions in the same directions.\n\n\n\n"
},

{
    "location": "lib/approximations.html#LazySets.Approximations.box_approximation_symmetric",
    "page": "Approximations",
    "title": "LazySets.Approximations.box_approximation_symmetric",
    "category": "Function",
    "text": "box_approximation_symmetric(X)\n\nOverapproximation of a set by a hyperrectangle which contains the origin.\n\nInput\n\nX – a lazy set\n\nOuptut\n\nH – a symmetric interval around the origin which tightly contains the given set\n\nAlgorithm\n\nThe center of the box is the origin, and the radius is obtained by computing the maximum value of the support function evaluated at the canonical directions.\n\n\n\n"
},

{
    "location": "lib/approximations.html#LazySets.Approximations.diameter_approximation",
    "page": "Approximations",
    "title": "LazySets.Approximations.diameter_approximation",
    "category": "Function",
    "text": "diameter_approximation(X)\n\nApproximate diameter of a given set.\n\nInput\n\nX – a lazy set\n\nAlgorithm\n\nThe diameter is bounded by twice the radius. This function relies on radius_approximation.\n\n\n\n"
},

{
    "location": "lib/approximations.html#LazySets.Approximations.radius_approximation",
    "page": "Approximations",
    "title": "LazySets.Approximations.radius_approximation",
    "category": "Function",
    "text": "radius_approximation(X)\n\nApproximate radius of a given set.\n\nInput\n\nX – a lazy set\n\nAlgorithm\n\nThis is an approximation in the infinity norm. The radius of a BallInf of center c and radius r can be approximated by ‖c‖ + r√n, where n is the dimension of the  vectorspace.\n\n\n\n"
},

{
    "location": "lib/approximations.html#Box-Approximations-1",
    "page": "Approximations",
    "title": "Box Approximations",
    "category": "section",
    "text": "ballinf_approximation\nbox_approximation\nbox_approximation_symmetric\ndiameter_approximation\nradius_approximation"
},

{
    "location": "lib/approximations.html#LazySets.jump2pi",
    "page": "Approximations",
    "title": "LazySets.jump2pi",
    "category": "Function",
    "text": "jump2pi(x)\n\nReturn x + 2 and only if x is negative.\n\nInput\n\nx – a floating point number\n\nExamples\n\njulia> jump2pi(0.0)\n0.0\njulia> jump2pi(-0.5)\n5.783185307179586\njulia> jump2pi(0.5)\n0.5\n\n\n\n"
},

{
    "location": "lib/approximations.html#Fast-2D-LPs-1",
    "page": "Approximations",
    "title": "Fast 2D LPs",
    "category": "section",
    "text": "Since vectors in the plane can be ordered by the angle with respect to the positive real axis, we can efficiently evaluate the support vector of a polygon in constraint representation by comparing normal directions, provided that its edges are ordered. We use the symbol preceq to compare directions, where the increasing direction is counter-clockwise.jump2pi(Image: ../assets/intuition2dlp.png)"
},

{
    "location": "about.html#",
    "page": "About",
    "title": "About",
    "category": "page",
    "text": ""
},

{
    "location": "about.html#Contributing-1",
    "page": "About",
    "title": "Contributing",
    "category": "section",
    "text": "Pages = [\"about/CONTRIBUTING.md\"]This page details the some of the guidelines that should be followed when contributing to this package."
},

{
    "location": "about.html#Running-the-Unit-Tests-1",
    "page": "About",
    "title": "Running the Unit Tests",
    "category": "section",
    "text": "$ julia --color=yes test/runtests.jl"
},

{
    "location": "about.html#Branches-1",
    "page": "About",
    "title": "Branches",
    "category": "section",
    "text": ""
},

{
    "location": "about.html#Contributing-to-the-Documentation-1",
    "page": "About",
    "title": "Contributing to the Documentation",
    "category": "section",
    "text": "The documentation source is written with Markdown, and we use Documenter.jl to produce the HTML documentation. To build the docs, run make.jl:$ julia --color=yes docs/make.jl"
},

{
    "location": "about.html#Credits-1",
    "page": "About",
    "title": "Credits",
    "category": "section",
    "text": "These persons have contributed to LazySets.jl (in alphabetic order):Marcelo Forets\nChristian Schilling\nFrederic ViryWe are also grateful to Goran Frehse for enlightening discussions."
},

]}
