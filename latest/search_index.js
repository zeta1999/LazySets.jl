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
    "text": "DocTestFilters = [r\"[0-9\\.]+ seconds \\(.*\\)\"]LazySets is a Julia package for calculus with convex sets.The aim is to provide a scalable library for solving complex set-based problems, such as those encountered in differential inclusions or reachability analysis techniques in the domain of formal verification. Typically, one is confronted with a set-based recurrence with a given initial set and/or input sets, and for visualization purposes the final result has to be obtained through an adequate projection onto low-dimensions. This library implements types to construct set formulas and methods to efficiently and accurately approximate the projection in low-dimensions.Pages = [\"index.md\"]"
},

{
    "location": "index.html#Introduction-1",
    "page": "Home",
    "title": "Introduction",
    "category": "section",
    "text": "In this package we describe convex sets lazily (i.e., symbolically). This way we provide an exact but abstract representation, in principle for any common convex set class or operation between sets. Concrete information is obtained through evaluating the set in specific directions. More precisely, each concrete subtype mathcalX of the abstract type LazySet exports a method to calculate its support vector sigma(d mathcalX) in a given (arbitrary) direction d in mathbbR^n. Representing sets exactly but lazily has the advantage of being able to perform only the required operations on-demand.For very long sequences of computations (e.g., set-based recurrences with tens of thousands of elements), it is useful to combine both lazy and concrete representations such as polyhedral approximations. All this is easy to do with LazySets. Moreover, we provide a specialized module for handling Cartesian decomposition of two-dimensional projections. The projection can be taken to the desired precision using an iterative refinement method."
},

{
    "location": "index.html#Example-1",
    "page": "Home",
    "title": "Example",
    "category": "section",
    "text": "Let mathcalX_0 subset mathbbR^1000 be the Euclidean ball of center (1 ldots 1) and radius 01 in dimension n=1000. Given a real matrix A in mathbbR^1000 times 1000, suppose that we are interested in the equationmathcalY = CH(e^A  mathcalX_0   BmathcalU mathcalX_0)where CH is the convex hull operator,  denotes Minkowski sum, mathcalU is a ball in the infinity norm centered at zero and radius 12, and B is a linear map of the appropriate dimensions. This equation typically arises in the study of discrete approximation models for reachability of continuous systems, see for example SpaceEx: Scalable verification of hybrid systems.For concreteness, we take A to be a random matrix with probability 1 of any entry being nonzero. Suppose that the input set mathcalU is two-dimensional, and that the linear map B is random. Finally, let δ = 0.1. Using LazySets, we can define this problem as follows:julia> using LazySets;\n\njulia> A = sprandn(1000, 1000, 0.01);\n\njulia> δ = 0.1;\n\njulia> X0 = Ball2(ones(1000), 0.1);\n\njulia> B = randn(1000, 2);\n\njulia> U = BallInf(zeros(2), 1.2);\nThe @time macro reveals that building mathcalY with LazySets is instantaneous:julia> @time Y = CH(SparseMatrixExp(A * δ) * X0 + δ * B * U, X0);\n0.000022 seconds (13 allocations: 16.094 KiB)By asking for the concrete type of Y, we see that it has a convex hull type, parameterized by the types of its arguments, corresponding to the mathematical formulation:julia> typeof(Y)\nLazySets.ConvexHull{LazySets.MinkowskiSum{LazySets.ExponentialMap{LazySets.Ball2{Float64}},LazySets.LinearMap{LazySets.BallInf{Float64},Float64}},LazySets.Ball2{Float64}}Now suppose that we are interested in observing the projection of mathcalY onto the variables number 1 and 500. First we define the 21000 projection matrix and apply it to mathcalY as a linear map (i.e., from the left). Second, we use the overapproximate method:julia> proj_mat = [[1. zeros(1, 999)]; [zeros(1, 499) 1. zeros(1, 500)]];\n\njulia> @time res = Approximations.overapproximate(proj_mat * Y);\n0.064034 seconds (1.12 k allocations: 7.691 MiB)We have calculated a box overapproximation of the exact projection onto the (x_1 x_500) plane. Notice that it takes about 0.064 seconds for the whole operation, allocating less than 10MB of RAM. Let us note that if the set operations were done explicitly, this would be much (!) slower. For instance, already the explicit computation of the matrix exponential would have cost 10x more, and allocated around 300MB. For even higher n, an evaluation will probably run out of RAM. But this is doable with LazySets because the action of the matrix exponential on the set is only evaluated along the directions of interest. Similar comments apply to the Minkowski sum above.We can visualize the result using plot, as shown below (left-most plot).(Image: assets/example_ch.png)In the second and third plots, we have used a refined method that allows to specify a prescribed accuracy for the projection (in terms of the Hausdorff distance). For the theoretical background, see this reference. It can be passed as a second argument to overapproximate.Error tol. time (s) memory (MB)\n∞ (no refinement) 0.022 5.27\n1e-1 0.051 7.91\n1e-3 0.17 30.3This table shows the runtime and memory consumption for different error tolerances, and the results are shown in three plots of above, from left to right. When passing to a smaller tolerance, the corners connecting edges are more \"rounded\", at the expense of computational resources, since more support vectors have to be evaluated."
},

{
    "location": "index.html#Features-1",
    "page": "Home",
    "title": "Features",
    "category": "section",
    "text": "The core functionality of LazySets is:Lazy (i.e., symbolic) types for several classes of convex sets such as balls in different norms, polygons in constraint or vertex representation, zonotopes, special types such as lines and linear constraints, hyperrectangles, and high-dimensional polyhedra.\nLazy implementations for most commonly used set operations, e.g., Minkowski sum, Cartesian product, convex hull and interval hull approximations, and linear and exponential maps.On top of the previous basic type representations and operations, LazySets can be used to:Efficiently evaluate the support vector of nested lazy sets.\nCartesian decomposition of lazy sets using two-dimensional projections.\nFast overapproximation of an exact set using a polyhedral approximation, to the desired accuracy.\nExtensive visualization capabilities through the Plots.jl framework."
},

{
    "location": "index.html#Manual-Outline-1",
    "page": "Home",
    "title": "Manual Outline",
    "category": "section",
    "text": "Pages = [\n    \"man/getting_started.md\",\n    \"man/polyhedral_approximations.md\",\n    \"man/decompose_example.md\",\n    \"man/fast_2d_LPs.md\",\n    \"man/iterative_refinement.md\"\n]\nDepth = 2"
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
    "text": "This package requires Julia v0.6 or later. Refer to the official documentation on how to install it for your system. Below we explain the steps for setting up LazySets on your system and checking that it builds correctly."
},

{
    "location": "man/getting_started.html#Installation-1",
    "page": "Getting Started",
    "title": "Installation",
    "category": "section",
    "text": "To install LazySets, use the following command inside Julia's REPL:Pkg.clone(\"https://github.com/JuliaReach/LazySets.jl\")The dependencies of LazySets, such as Expokit.jl – which provides lazy matrix exponentiation routines – are automatically installed through Julia's package manager. The full list of dependencies is specified in the REQUIRE file."
},

{
    "location": "man/getting_started.html#Workflow-tips-1",
    "page": "Getting Started",
    "title": "Workflow tips",
    "category": "section",
    "text": "There are different ways to use Julia: from the terminal, from the Julia REPL, from IJulia (i.e., Jupyter notebook), from Juno, etc. If you do not have a preferred choice, we recommend using LazySets through IJulia; one reason is that the visualization is conveniently embedded into the notebook, and it can be exported into different formats, among other benefits. On the other hand, for development purposes you probably want to use the REPL or the Juno environment."
},

{
    "location": "man/getting_started.html#Updating-1",
    "page": "Getting Started",
    "title": "Updating",
    "category": "section",
    "text": "After working with LazySets for some time, you may want to get the newest version. For this you can use the following command (e.g., from the REPL):Pkg.checkout(\"LazySets\")That will check out the latest version in the master branch, and precompile it the next time you enter a session and execute using LazySets."
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
    "text": "The support function is a basic notion for approximating convex sets. Let mathcalX subset mathbbR^n be a compact convex set. The support function of mathcalX is the function rho_mathcalX  mathbbR^nto mathbbR, defined asrho_mathcalX(ell) = maxlimits_x in mathcalX ell^mathrmT xWe recall the following elementary properties of the support function.Proposition. For all compact convex sets mathcalX, mathcalY in mathbbR^n, for all ntimes n real matrices M, all scalars lambda, and all vectors ell in mathbbR^n, we have:beginalign*\nquad rho_lambdamathcalX (ell) = rho_mathcalX (lambda ell)\ntext and  rho_lambdamathcalX (ell) = lambda rho_mathcalX (ell) text if  lambda  0 tag11 1mm\n\nquad rho_MmathcalX (ell) = rho_mathcalX (M^mathrmT ell) tag12 1mm\n\nquad rho_mathcalX oplus mathcalY (ell) = rho_mathcalX (ell) + rho_mathcalY (ell) tag13 1mm\n\nquad rho_mathcalX times mathcalY (ell) = ell^mathrmT sigma_mathcalX times mathcalY(ell) tag14 1mm\n\nquad rho_mathrmCH(mathcalXcupmathcalY) (ell) = max (rho_mathcalX (ell) rho_mathcalY (ell)) tag15\nendalign*"
},

{
    "location": "man/polyhedral_approximations.html#Support-Vector-1",
    "page": "Polyhedral Approximations",
    "title": "Support Vector",
    "category": "section",
    "text": "The farthest points of mathcalX in the direction ell  are the support vectors denoted sigma_mathcalX(ell). These points correspond to the optimal points for the support function, i.e.,sigma_mathcalX(ell) =  x in mathcalX  ell^mathrmT x  = rho_mathcalX(ell)  Since all support vectors in a given direction evaluate to the same value of the support function, we often speak of the support vector, where the choice of any support vector is implied.(Image: Illustration of the support function and the support vector)Proposition 2. Under the same conditions as in Proposition 1, the following hold:beginalign*\nquad sigma_lambdamathcalX (ell) = lambda sigma_mathcalX (lambda ell) tag21 1mm\n\nquad sigma_MmathcalX (ell) = Msigma_mathcalX (M^mathrmT ell) tag22 1mm\n\nquad sigma_mathcalX oplus mathcalY (ell) = sigma_mathcalX (ell) oplus sigma_mathcalY (ell) tag23 1mm\n\nquad sigma_mathcalX times mathcalY (ell) = (sigma_mathcalX(ell_1) sigma_mathcalY(ell_2)) text where  ell = (ell_1 ell_2) tag24 1mm\n\nquad sigma_mathrmCH(mathcalXcupmathcalY) (ell) =\ntextargmax_x y (ell^mathrmT x ell^mathrmT y)\ntext where  x in sigma_mathcalX(ell) y in sigma_mathcalY(ell) tag25\nendalign*"
},

{
    "location": "man/polyhedral_approximations.html#Polyhedral-approximation-of-a-convex-set-1",
    "page": "Polyhedral Approximations",
    "title": "Polyhedral approximation of a convex set",
    "category": "section",
    "text": "The projection of a set into a low dimensional space (a special case of M mathcalX) can be conveniently evaluated using support functions, since sigma_MmathcalX(ell) = sigma_mathcalX(M^Tell). Moreover, for some classical convex sets such as unit balls in the infinity norm, in the 2-norm, or polyhedra in constraint representation, the support functions can be efficiently computed. For example, the support function of the unit ball mathcalB_p^n is rho_mathcalB_p^n(ell) = VertellVert_fracpp-1Given directions ell_1ldotsell_m, a tight overapproximation of mathcalX is the outer polyhedron given by the constraints beginequation*\nquad bigwedge_i ell_i^T x leq rho_mathcalX(ell_i) tag3\nendequation*For instance, a bounding box involves evaluating the support function in 2n directions. To quantify this, we use the following distance measure.A set mathcalhatX is within Hausdorff distance varepsilon of mathcalX if and only ifmathcalhatX subseteq mathcalX oplus varepsilonmathcalB_p^n\ntext and  mathcalX subseteq mathcalhatX oplus\nvarepsilonmathcalB_p^nThe infimum varepsilon geq 0 that satisfies the above equation is called the Hausdorff distance between mathcalX and mathcalhatX with respect to the p-norm, and is denoted d_H^pbigl(mathcalXmathcalhatXbigr).Another useful characterization of the Hausdorff distance is the following. Let mathcalX mathcalY subset mathbbR^n be polytopes. Thend^p_H(mathcalX mathcalY) = max_ell in mathcalB_p^n\nleftrho_mathcalY(ell) - rho_mathcalX(ell)rightIn the special case mathcalX subseteq mathcalY, the absolute value can be removed.By adding directions using Lotov's method (s. below), the outer polyhedron in (3) is within Hausdorff distance varepsilon VertXVert_p for mathcalOleft(frac1varepsilon^n-1right) directions, and this bound is optimal. It follows that accurate outer polyhedral approximations are possible only in low dimensions. For n=2, the bound can be lowered to mathcalOleft(frac1sqrtvarepsilonright) directions, which is particularly efficient and the reason why we chose to decompose the system into subsystems of dimension 2."
},

{
    "location": "man/polyhedral_approximations.html#Lotov's-method-1",
    "page": "Polyhedral Approximations",
    "title": "Lotov's method",
    "category": "section",
    "text": "An overapproximation of the projections of a polyhedron given in constraint form can be obtained using Lotov's method; this is a particularly effective method in two dimensions. Lotov's algorithm proceeds as follows. Starting with at least n linearly independent template directions, compute an outer approximation. From the corresponding support vectors, compute an inner approximation, as the convex hull of the support vectors. Now compute the facet normals of the inner approximation, and the distance between the facets of the inner and the vertices of the outer approximation. Finally, pick the facet normal with the largest distance, and add it to the template directions. This procedure is repeated until the distance is smaller than the desired error.For more details we refer to the paper."
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
    "text": "In this section we present an illustrative example of the decomposed image of a linear map.Pages = [\"decompose_example.md\"]\nDepth = 3"
},

{
    "location": "man/decompose_example.html#Preliminaries:-Polygon,-Linear-Map,-and-Plotting-1",
    "page": "Decomposing an Affine Map",
    "title": "Preliminaries: Polygon, Linear Map, and Plotting",
    "category": "section",
    "text": "Consider the matrix-valued function () = beginpmatrix cos ()  -sin ()  sin ()  cos () endpmatrix,    .using LazySets, LazySets.Approximations, Plots\n\nΦ(theta) = [cos(theta) -sin(theta); sin(theta) cos(theta)]Now define an arbitrary convex polygon with five vertices with operatornameCH denoting the convex hull operation,mathcalX = operatornameCHbig( (1 05) (11 02) (14 03) (17 05) (14 08) big)This set can be defined as:X = VPolygon([[1.0, 0.5], [1.1, 0.2], [1.4, 0.3], [1.7, 0.5], [1.4, 0.8]])note: Note\nYou can as well define the convex hull of the one element sets (singletons) viaC = CH([Singleton([1.0, 0.5]), Singleton([1.1, 0.2]), Singleton([1.4, 0.3]), Singleton([1.7, 0.5]), Singleton([1.4, 0.8])])Observe that C is just a lazy convex hull, whereas X is a polygon in vertex representation.Applying the linear map (4)  mathcalX, we get a new polygon mathcalX which is the counter-clockwise turn of mathcalX by  triangleq 45. In this package the linear map is not computed explicitly but only wrapped in a LinearMap instance.Xp = Φ(pi/4) * X\n\ntypeof(Xp)Let us plot the two polygons, mathcalX in green and mathcalX in blue.example = plot(X, color=\"green\")\n\nplot!(example, Xp, 1e-2, color=\"blue\")Note that we have passed 1e-2 as additional argument for the LinearMap set (mathcalX) because by default such a set is just plotted as its box (or hyperrectangle) approximation. The value 1e-2 is the precision up to which the set is (over-)approximated with a polgon, which in this case is sufficient to obtain the actual set again."
},

{
    "location": "man/decompose_example.html#Cartesian-Decomposition-1",
    "page": "Decomposing an Affine Map",
    "title": "Cartesian Decomposition",
    "category": "section",
    "text": "Next we want to decompose mathcalX into a Cartesian product of intervals. That is, we project it to the x-axis and y-axis and then compose these intervals again: hatmathcalX = hatmathcalX_1 times hatmathcalX_2.Xhat = overapproximate(X)  # approximation of X with an axis-aligned polygon\n\nplot!(example, Xhat, color=\"gray\", alpha=0.3)"
},

{
    "location": "man/decompose_example.html#Decomposed-Image-of-a-Linear-Map-1",
    "page": "Decomposing an Affine Map",
    "title": "Decomposed Image of a Linear Map",
    "category": "section",
    "text": "Now let us compute the linear map for the box approximation, and let us call it mathcalY = (4)  hatmathcalX. This will be a diamond-like shape (the box turned by 45°).Y = Φ(pi/4) * Xhat\n\nplot!(example, Y, 1e-2, color=\"yellow\", alpha=0.3)However, we want our approximation be again a Cartesian product of intervals, so we have to overapproximate this diamond-like shape again: hatmathcalY = hatmathcalX = hatmathcalX_1 times hatmathcalX_2Xhatp = overapproximate(Y)\n\nplot!(example, Xhatp, 1e-2, color=\"gray\", alpha=0.3)As we can see, the resulting box hatmathcalX is not a tight overapproximation of mathcalX. We can, however, gain precision by reducing the angle by which we turn the set, e.g., making two smaller turns. Why not try it out?"
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
    "text": "Since vectors in the plane can be ordered by the angle with respect to the positive real axis, we can efficiently evaluate the support vector of a polygon in constraint representation by comparing normal directions, provided that its edges are ordered.This is illustrated in the following picture.(Image: ../assets/intuition2dlp.png)If the normal directions of the polygon are ordered, the support vector in any direction always lies between two consecutive edges, a_i+1 preceq ell preceq a_i. Here we use the symbol preceq to compare directions, where the increasing direction is counter-clockwise.The following lemma provides an algorithm to find the support vector."
},

{
    "location": "man/fast_2d_LPs.html#Lemma-1",
    "page": "Fast 2D LPs",
    "title": "Lemma",
    "category": "section",
    "text": "Let mathcalX be a polygon described by m linear constraints a_i^T x leq b_i, ordered by the normal vectors (a_i), i.e., a_i preceq a_i+1 for all i in 1ldotsm, where we identify a_m+1 with a_1. Let ell in mathbbR^2 setminus mathbf0_2. Then there exists i in 1dotsm such that a_i preceq ell preceq a_i+1 and every optimal solution barx of the linear program rho_mathcalX(ell) = max ell^T x  x in mathcalX satisfies barx in x  a_i^T x leq b_i cap x  a_i+1^T x leq b_i+1"
},

{
    "location": "man/fast_2d_LPs.html#Algorithm-1",
    "page": "Fast 2D LPs",
    "title": "Algorithm",
    "category": "section",
    "text": "First define a <= b as the comparison of directions using polar angles, with 0 being the direction (1, 0).Now assume that the constraints in a polytope mathcalX are given as a_i x + b_i.The following pseudocode explains how to find barx.σ(d, X):\n    let i be the smallest index such that a_{i-1} <= d and a_i > d\n    return the vertex at the intersection of constraints i and i-1"
},

{
    "location": "man/iterative_refinement.html#",
    "page": "Iterative refinement",
    "title": "Iterative refinement",
    "category": "page",
    "text": ""
},

{
    "location": "man/iterative_refinement.html#LazySets.Approximations.Approximation2D",
    "page": "Iterative refinement",
    "title": "LazySets.Approximations.Approximation2D",
    "category": "Type",
    "text": "Approximation2D{N<:AbstractFloat}\n\nType that represents a local approximation in 2D.\n\nFields\n\np1        – first inner point\nd1        – first direction\np2        – second inner point\nd2        – second direction\nerr       – error made\nndir      – normal direction of the inner approximation\nrefinable – states if this approximation is refinable\n\n\n\n"
},

{
    "location": "man/iterative_refinement.html#LazySets.Approximations.Approximation2D-NTuple{4,Array{Float64,1}}",
    "page": "Iterative refinement",
    "title": "LazySets.Approximations.Approximation2D",
    "category": "Method",
    "text": "Approximation2D(p1::Vector{N}, d1::Vector{N}, p2::Vector{N}, d2::Vector{N}) where {N<:AbstractFloat}\n\nConstructor of Approximation2D from two inner points and two directions.\n\nInput\n\np1        – first inner point\nd1        – first direction\np2        – second inner point\nd2        – second direction\n\nOutput\n\nA new Approximation2D instance.\n\n\n\n"
},

{
    "location": "man/iterative_refinement.html#LazySets.Approximations.refine-Tuple{LazySets.LazySet,LazySets.Approximations.Approximation2D}",
    "page": "Iterative refinement",
    "title": "LazySets.Approximations.refine",
    "category": "Method",
    "text": "refine(S::LazySet, approx::Approximation2D)::Tuple{Approximation2D, Approximation2D}\n\nRefine the given approximation.\n\nInput\n\nS      – 2D convex set that is approximated\napprox – approximation to refine\n\nOutput\n\nThe refined approximation.\n\n\n\n"
},

{
    "location": "man/iterative_refinement.html#Iterative-Refinement-1",
    "page": "Iterative refinement",
    "title": "Iterative Refinement",
    "category": "section",
    "text": "This section of the manual describes an approximation method for an arbitrary two-dimensional convex set S and a given error bound  using support vectors.CurrentModule = LazySets.ApproximationsThe basic idea is to add new supporting directions whenever the approximation error is still bigger than .The approximation is represented by a list of local refinements. Each refinement describes a set with one angle and is wrapped in the following type.Approximation2D\nApproximation2D(::Vector{Float64}, ::Vector{Float64}, ::Vector{Float64}, ::Vector{Float64})The approximation is initialized with box directions, i.e., we have four refinement instances, one for each angle. Then we just iterate through all refinement instances and check if the error is bigger than the threshold individually. If so, we refine the instance by splitting into two more precise refinement instances and apply the checks recursively.refine(::LazySet, ::Approximation2D)"
},

{
    "location": "man/interval_hulls.html#",
    "page": "Interval Hulls",
    "title": "Interval Hulls",
    "category": "page",
    "text": ""
},

{
    "location": "man/interval_hulls.html#Interval-Hulls-1",
    "page": "Interval Hulls",
    "title": "Interval Hulls",
    "category": "section",
    "text": "In this section we illustrate the interval hull operators as well as several plotting functionalities.Pages = [\"interval_hulls.md\"]\nDepth = 3DocTestSetup = quote\n    using LazySets, Plots, LazySets.Approximations\nend"
},

{
    "location": "man/interval_hulls.html#Balls-and-Singletons-1",
    "page": "Interval Hulls",
    "title": "Balls and Singletons",
    "category": "section",
    "text": "Consider a ball in the 2-norm. By default, the coefficients of this set are 64-bit floating point numbers. Other numeric types (such as lower precision floating point, or rational) can be defined with the proper argument types in the Ball2 constructor.using LazySets, Plots\n\nX = Ball2(ones(2), 0.5)To plot a lazy set, we use the plot function. By design, lazy sets plots overapproximate with box directions only. To have a sharp definition of the borders, use the accuracy as a second argument.plot(X, 1e-3, aspectratio=1)To add plots to the same pair of axes we use plot!. Let's add some points of the set which are farthest in some given directions. Single points can be plotted using the Singleton type. In the third line of code we plot the center of the ball picking a custom cross marker.plot!(Singleton(σ([1., 0], X)))\nplot!(Singleton(σ([1., 1], X)))\nplot!(Singleton(X.center), markershape=:x)note: Note\nTo see the list of available plot keyword arguments, use the plotattr([attr]) function, where attr is the symbol :Plot, :Series, :Axis or :Subplot.For the remainder of this section we define another ball in the 2-norm and its convex hull with X.Y = Ball2([-3,-.5], 0.8)\nZ = CH(X, Y)\n\nplot(X, 1e-3, aspectratio=1)\nplot!(Y, 1e-3)\nplot!(Z, 1e-3, alpha=0.2)"
},

{
    "location": "man/interval_hulls.html#Ballinf-approximation-1",
    "page": "Interval Hulls",
    "title": "Ballinf approximation",
    "category": "section",
    "text": "A simple overapproximation with a BallInf is obtained with the ballinf_approximation function, from the Approximations module. It overapproximates a convex set by a tight ball in the infinity norm by evaluating the support vector in the canonical directions.import LazySets.Approximations.ballinf_approximation\n\nplot(X, 1e-3, aspectratio=1)\nplot!(Y, 1e-3)\nplot!(Z, 1e-3, alpha=0.2)\n\nBapprox = ballinf_approximation(Z)\n\nplot!(Bapprox, alpha=0.1)\nplot!(Singleton(Bapprox.center), markershape=:x)Bapprox.center, Bapprox.radius"
},

{
    "location": "man/interval_hulls.html#Interval-hull-approximation-1",
    "page": "Interval Hulls",
    "title": "Interval hull approximation",
    "category": "section",
    "text": "If we want to have different lengths for each dimension, instead of the ballinf_approximation, we can use the approximation with a hyperrectangle through the interval_hull function.import LazySets.Approximations.interval_hull\n\nplot(X, 1e-3, aspectratio=1)\nplot!(Y, 1e-3)\nplot!(Z, 1e-3, alpha=0.2)\n\nHapprox = interval_hull(Z)\n\nplot!(Happrox, alpha=0.1)\nplot!(Singleton(Happrox.center), markershape=:x)Happrox.center, Happrox.radiusnote: Note\nThe interval_hull function is an alias for the box_approximation function. The nomenclature for approximation functions is *_approximation_*. To see a list of all approximation functions, either search in the docs or type names(LazySets.Approximations)."
},

{
    "location": "man/interval_hulls.html#Symmetric-interval-hull-1",
    "page": "Interval Hulls",
    "title": "Symmetric interval hull",
    "category": "section",
    "text": "Contrary to the previous approximations, the symmetric interval hull is centered around the origin. It is defined in the Approximations module as well.import LazySets.Approximations.symmetric_interval_hull\n\nplot(X, 1e-3, aspectratio=1)\nplot!(Y, 1e-3)\nplot!(Z, 1e-3, alpha=0.2)\n\nS = symmetric_interval_hull(Z)\nplot!(S, alpha=0.2)\nplot!(Singleton(S.center), markershape=:x)S.center, S.radiusWe can get the list of vertices using the vertices_list function:vertices_list(S)For instance, compute the support vector in the south-east direction:σ([1., -1.], S)It is also possible to pass a sparse vector as direction, and the result is a  sparse vector:σ(sparsevec([1., -1.]), S)"
},

{
    "location": "man/interval_hulls.html#Norm,-radius-and-diameter-1",
    "page": "Interval Hulls",
    "title": "Norm, radius and diameter",
    "category": "section",
    "text": "In this part we illustrate some functions to obtain metric properties of sets, applied to the sets X, Y and Z defined previously, in the infinity norm. These functions apply generally to any LazySet. For some types, specialized methods are triggered automatically through multiple-dispatch.The norm of a convex set is the norm of the enclosing ball (of the given norm) of minimal volume. For instance:import LazySets.Approximations: norm, radius, diameter\n\nnorm(X), norm(Y), norm(Z)The radius of a convex set. It is the radius of the enclosing ball (of the given norm) of minimal volume with the same center. In the previous example,radius(X), radius(Y), radius(Z)Finally, it is sometimes convenient to ask directly the diameter of the set, defined as twice the radius:diameter(X), diameter(Y), diameter(Z)"
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
    "text": "This section of the manual describes the basic set representation types.Pages = [\"representations.md\"]\nDepth = 3CurrentModule = LazySets\nDocTestSetup = quote\n    using LazySets\nend"
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
    "text": "LazySet\n\nAbstract type for a lazy set.\n\nNotes\n\nEvery concrete LazySet must define a function σ(d, X), representing the support vector of X in a given direction d, and dim, the ambient dimension of the set X.\n\nLazySet types should be parameterized with a type N, typically N<:Real, to support computations with different numeric types.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.ρ",
    "page": "Common Set Representations",
    "title": "LazySets.ρ",
    "category": "Function",
    "text": "ρ(d::AbstractVector{N}, S::LazySet)::N where {N<:Real}\n\nEvaluate the support function of a set in a given direction.\n\nInput\n\nd – direction\nS – convex set\n\nOutput\n\nThe support function of the set S for the direction d.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.support_function",
    "page": "Common Set Representations",
    "title": "LazySets.support_function",
    "category": "Function",
    "text": "support_function\n\nAlias for the support function ρ.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.support_vector",
    "page": "Common Set Representations",
    "title": "LazySets.support_vector",
    "category": "Function",
    "text": "support_vector\n\nAlias for the support vector σ.\n\n\n\n"
},

{
    "location": "lib/representations.html#Abstract-support-function-and-support-vector-1",
    "page": "Common Set Representations",
    "title": "Abstract support function and support vector",
    "category": "section",
    "text": "LazySets\nLazySets.LazySet\nρ\nsupport_function\nsupport_vector"
},

{
    "location": "lib/representations.html#Balls-1",
    "page": "Common Set Representations",
    "title": "Balls",
    "category": "section",
    "text": ""
},

{
    "location": "lib/representations.html#LazySets.Ball2",
    "page": "Common Set Representations",
    "title": "LazySets.Ball2",
    "category": "Type",
    "text": "Ball2{N<:Real} <: LazySet\n\nType that represents a ball in the 2-norm.\n\nFields\n\ncenter – center of the ball as a real vector\nradius – radius of the ball as a real scalar ( 0)\n\nNotes\n\nMathematically, a ball in the 2-norm is defined as the set\n\nmathcalB_2^n(c r) =  x  mathbbR^n   x - c _2  r \n\nwhere c  mathbbR^n is its center and r  mathbbR_+ its radius. Here   _2 denotes the Euclidean norm (also known as 2-norm), defined as  x _2 = left( sumlimits_i=1^n x_i^2 right)^12 for any x  mathbbR^n.\n\nExamples\n\nCreate a five-dimensional ball B in the 2-norm centered at the origin with radius 0.5:\n\njulia> B = Ball2(zeros(5), 0.5)\nLazySets.Ball2{Float64}([0.0, 0.0, 0.0, 0.0, 0.0], 0.5)\njulia> dim(B)\n5\n\nEvaluate B's support vector in the direction 12345:\n\njulia> σ([1.,2.,3.,4.,5.], B)\n5-element Array{Float64,1}:\n 0.06742\n 0.13484\n 0.20226\n 0.26968\n 0.3371\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.dim-Tuple{LazySets.Ball2}",
    "page": "Common Set Representations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(B::Ball2)::Int\n\nReturn the dimension of a ball in the 2-norm.\n\nInput\n\nB – ball in the 2-norm\n\nOutput\n\nThe ambient dimension of the ball.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.Ball2}",
    "page": "Common Set Representations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d::AbstractVector{N}, B::Ball2)::AbstractVector{<:AbstractFloat} where {N<:AbstractFloat}\n\nReturn the support vector of a 2-norm ball in a given direction.\n\nInput\n\nd – direction\nB – ball in the 2-norm\n\nOutput\n\nThe support vector in the given direction. If the direction has norm zero, the origin is returned.\n\nNotes\n\nLet c and r be the center and radius of a ball B in the 2-norm, respectively. For nonzero direction d we have (d B) = c + r * (d  d_2).\n\nThis function requires computing the 2-norm of the input direction, which is performed in the given precision of the numeric datatype of both the direction and the set. Exact inputs are not supported.\n\n\n\n"
},

{
    "location": "lib/representations.html#Base.:∈-Tuple{AbstractArray{Float64,1},LazySets.Ball2{Float64}}",
    "page": "Common Set Representations",
    "title": "Base.:∈",
    "category": "Method",
    "text": "∈(x::AbstractVector{N}, B::Ball2{N})::Bool where {N<:Real}\n\nCheck whether a given point is contained in a ball in the 2-norm.\n\nInput\n\nx – point/vector\nB – ball in the 2-norm\n\nOutput\n\ntrue iff x  B.\n\nNotes\n\nThis implementation is worst-case optimized, i.e., it is optimistic and first computes (see below) the whole sum before comparing to the radius. In applications where the point is typically far away from the ball, a fail-fast implementation with interleaved comparisons could be more efficient.\n\nAlgorithm\n\nLet B be an n-dimensional ball in the 2-norm with radius r and let c_i and x_i be the ball's center and the vector x in dimension i, respectively. Then x  B iff left( _i=1^n c_i - x_i^2 right)^12  r.\n\nExamples\n\njulia> B = Ball2([1., 1.], sqrt(0.5))\nLazySets.Ball2{Float64}([1.0, 1.0], 0.7071067811865476)\njulia> ∈([.5, 1.6], B)\nfalse\njulia> ∈([.5, 1.5], B)\ntrue\n\n\n\n"
},

{
    "location": "lib/representations.html#Euclidean-norm-ball-1",
    "page": "Common Set Representations",
    "title": "Euclidean norm ball",
    "category": "section",
    "text": "Ball2\ndim(::Ball2)\nσ(::AbstractVector{Float64}, ::Ball2)\n∈(::AbstractVector{Float64}, ::Ball2{Float64})"
},

{
    "location": "lib/representations.html#LazySets.BallInf",
    "page": "Common Set Representations",
    "title": "LazySets.BallInf",
    "category": "Type",
    "text": "BallInf{N<:Real} <: LazySet\n\nType that represents a ball in the infinity norm.\n\nFields\n\ncenter – center of the ball as a real vector\nradius – radius of the ball as a real scalar ( 0)\n\nNotes\n\nMathematically, a ball in the infinity norm is defined as the set\n\nmathcalB_^n(c r) =  x  mathbbR^n   x - c _  r \n\nwhere c  mathbbR^n is its center and r  mathbbR_+ its radius. Here   _ denotes the infinity norm, defined as  x _ = maxlimits_i=1n vert x_i vert for any x  mathbbR^n.\n\nExamples\n\nCreate the two-dimensional unit ball and compute its support function along the positive x=y direction:\n\njulia> B = BallInf(zeros(2), 1.0)\nLazySets.BallInf{Float64}([0.0, 0.0], 1.0)\njulia> dim(B)\n2\njulia> ρ([1., 1.], B)\n2.0\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.dim-Tuple{LazySets.BallInf}",
    "page": "Common Set Representations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(B::BallInf)::Int\n\nReturn the dimension of a ball in the infinity norm.\n\nInput\n\nB – ball in the infinity norm\n\nOutput\n\nThe ambient dimension of the ball.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.BallInf}",
    "page": "Common Set Representations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d::AbstractVector{<:Real}, B::BallInf)::AbstractVector{<:Real}\n\nReturn the support vector of an infinity norm ball in a given direction.\n\nInput\n\nd – direction\nB – ball in the infinity norm\n\nOutput\n\nThe support vector in the given direction. If the direction has norm zero, the vertex with biggest values is returned.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.vertices_list-Tuple{LazySets.BallInf}",
    "page": "Common Set Representations",
    "title": "LazySets.vertices_list",
    "category": "Method",
    "text": "vertices_list(B::BallInf{N})::Vector{Vector{N}} where {N<:Real}\n\nReturn the list of vertices of a ball in the infinity norm.\n\nInput\n\nB – ball in the infinity norm\n\nOutput\n\nA list of vertices.\n\nNotes\n\nFor high dimensions, it is preferable to develop a vertex_iterator approach.\n\nExamples\n\njulia> B = BallInf(zeros(2), 0.1)\nLazySets.BallInf{Float64}([0.0, 0.0], 0.1)\njulia> vertices_list(B)\n4-element Array{Array{Float64,1},1}:\n [0.1, 0.1]\n [-0.1, 0.1]\n [0.1, -0.1]\n [-0.1, -0.1]\n\n\n\n"
},

{
    "location": "lib/representations.html#Base.LinAlg.norm",
    "page": "Common Set Representations",
    "title": "Base.LinAlg.norm",
    "category": "Function",
    "text": "norm(B::BallInf, [p]::Real=Inf)::Real\n\nReturn the norm of a ball in the infinity norm.\n\nInput\n\nB – ball in the infinity norm\np – (optional, default: Inf) norm\n\nOutput\n\nA real number representing the norm.\n\nNotes\n\nThe norm of an infinity ball is defined as the norm of the enclosing ball, of the given p-norm, of minimal volume.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.radius",
    "page": "Common Set Representations",
    "title": "LazySets.radius",
    "category": "Function",
    "text": "radius(B::BallInf, [p]::Real=Inf)::Real\n\nReturn the radius of a ball in the infinity norm.\n\nInput\n\nB – ball in the infinity norm\np – (optional, default: Inf) norm\n\nOutput\n\nA real number representing the radius.\n\nNotes\n\nThe radius is defined as the radius of the enclosing ball of the given p-norm of minimal volume with the same center.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.diameter",
    "page": "Common Set Representations",
    "title": "LazySets.diameter",
    "category": "Function",
    "text": "diameter(B::BallInf, [p]::Real=Inf)::Real\n\nReturn the diameter of a ball in the infinity norm.\n\nInput\n\nB – ball in the infinity norm\np – (optional, default: Inf) norm\n\nOutput\n\nA real number representing the diameter.\n\nNotes\n\nThe diameter is defined as the maximum distance in the given p-norm between any two elements of the set. Equivalently, it is the diameter of the enclosing ball of the given p-norm of minimal volume with the same center.\n\n\n\n"
},

{
    "location": "lib/representations.html#Base.:∈-Tuple{AbstractArray{Float64,1},LazySets.BallInf{Float64}}",
    "page": "Common Set Representations",
    "title": "Base.:∈",
    "category": "Method",
    "text": "∈(x::AbstractVector{N}, B::BallInf{N})::Bool where {N<:Real}\n\nCheck whether a given point is contained in a ball in the infinity norm.\n\nInput\n\nx – point/vector\nB – ball in the infinity norm\n\nOutput\n\ntrue iff x  B.\n\nAlgorithm\n\nLet B be an n-dimensional ball in the infinity norm with radius r and let c_i and x_i be the ball's center and the vector x in dimension i, respectively. Then x  B iff c_i - x_i  r for all i=1n.\n\nExamples\n\njulia> B = BallInf([1., 1.], 1.);\n\njulia> ∈([.5, -.5], B)\nfalse\njulia> ∈([.5, 1.5], B)\ntrue\n\n\n\n"
},

{
    "location": "lib/representations.html#Infinity-norm-ball-1",
    "page": "Common Set Representations",
    "title": "Infinity norm ball",
    "category": "section",
    "text": "BallInf\ndim(::BallInf)\nσ(::AbstractVector{Float64}, ::BallInf)\nvertices_list(::BallInf)\nnorm(::BallInf, ::Real=Inf)\nradius(::BallInf, ::Real=Inf)\ndiameter(::BallInf, ::Real=Inf)\n∈(::AbstractVector{Float64}, ::BallInf{Float64})"
},

{
    "location": "lib/representations.html#LazySets.Ball1",
    "page": "Common Set Representations",
    "title": "LazySets.Ball1",
    "category": "Type",
    "text": "Ball1 <: LazySet\n\nType that represents a ball in the 1-norm, also known as Manhattan or Taxicab norm.\n\nIt is defined as the set\n\nmathcalB_1^n(c r) =  x  mathbbR^n  _i=1^n c_i - x_i  r \n\nwhere c  mathbbR^n is its center and r  mathbbR_+ its radius.\n\nFields\n\ncenter – center of the ball as a real vector\nradius – radius of the ball as a scalar ( 0)\n\nExamples\n\nUnit ball in the 1-norm in the plane:\n\njulia> B = Ball1(zeros(2), 1.)\nLazySets.Ball1{Float64}([0.0, 0.0], 1.0)\njulia> dim(B)\n2\n\nWe evaluate the support vector in the East direction:\n\njulia> σ([0.,1], B)\n2-element Array{Float64,1}:\n 0.0\n 1.0\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.dim-Tuple{LazySets.Ball1}",
    "page": "Common Set Representations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(B::Ball1)::Int\n\nReturn the dimension of a Ball1.\n\nInput\n\nB – a ball in the 1-norm\n\nOutput\n\nThe ambient dimension of the ball.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.Ball1}",
    "page": "Common Set Representations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d::AbstractVector{N}, B::Ball1)::AbstractVector{N} where {N<:Real}\n\nReturn the support vector of a Ball1 in a given direction.\n\nInput\n\nd – a direction\nB – a ball in the p-norm\n\nOutput\n\nSupport vector in the given direction.\n\n\n\n"
},

{
    "location": "lib/representations.html#Base.:∈-Tuple{AbstractArray{Float64,1},LazySets.Ball1{Float64}}",
    "page": "Common Set Representations",
    "title": "Base.:∈",
    "category": "Method",
    "text": "∈(x::AbstractVector{N}, B::Ball1{N})::Bool where {N<:Real}\n\nCheck whether a given point is contained in a ball in the 1-norm.\n\nInput\n\nx – point/vector\nB – ball in the 1-norm\n\nOutput\n\ntrue iff x  B.\n\nNotes\n\nThis implementation is worst-case optimized, i.e., it is optimistic and first computes (see below) the whole sum before comparing to the radius. In applications where the point is typically far away from the ball, a fail-fast implementation with interleaved comparisons could be more efficient.\n\nAlgorithm\n\nLet B be an n-dimensional ball in the 1-norm with radius r and let c_i and x_i be the ball's center and the vector x in dimension i, respectively. Then x  B iff _i=1^n c_i - x_i  r.\n\nExamples\n\njulia> B = Ball1([1., 1.], 1.);\n\njulia> ∈([.5, -.5], B)\nfalse\njulia> ∈([.5, 1.5], B)\ntrue\n\n\n\n"
},

{
    "location": "lib/representations.html#Manhattan-norm-ball-1",
    "page": "Common Set Representations",
    "title": "Manhattan norm ball",
    "category": "section",
    "text": "Ball1\ndim(::Ball1)\nσ(::AbstractVector{Float64}, ::Ball1)\n∈(::AbstractVector{Float64}, ::Ball1{Float64})"
},

{
    "location": "lib/representations.html#LazySets.Ballp",
    "page": "Common Set Representations",
    "title": "LazySets.Ballp",
    "category": "Type",
    "text": "Ballp <: LazySet\n\nType that represents a ball in the p-norm, for 1  p  .\n\nIt is defined as the set\n\nmathcalB_p^n(c r) =  x  mathbbR^n   x - c _p  r \n\nwhere c  mathbbR^n is its center and r  mathbbR_+ its radius. Here   _p for 1  p   denotes the vector p-norm, defined as  x _p = left( sumlimits_i=1^n x_i^p right)^1p for any x  mathbbR^n.\n\nFields\n\np      – norm as a real scalar\ncenter – center of the ball as a real vector\nradius – radius of the ball as a scalar ( 0)\n\nNotes\n\nThe special cases p=1, p=2 and p= fall back to the specialized types Ball1, Ball2 and BallInf, respectively.\n\nExamples\n\nA five-dimensional ball in the p=32 norm centered at the origin of radius 0.5:\n\njulia> B = Ballp(3/2, zeros(5), 0.5)\nLazySets.Ballp{Float64}(1.5, [0.0, 0.0, 0.0, 0.0, 0.0], 0.5)\njulia> dim(B)\n5\n\nWe evaluate the support vector in direction 125:\n\njulia> σ(1.:5, B)\n5-element Array{Float64,1}:\n 0.013516\n 0.054064\n 0.121644\n 0.216256\n 0.3379\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.dim-Tuple{LazySets.Ballp}",
    "page": "Common Set Representations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(B::Ballp)::Int\n\nReturn the dimension of a Ballp.\n\nInput\n\nB – a ball in the p-norm\n\nOutput\n\nThe ambient dimension of the ball.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.Ballp}",
    "page": "Common Set Representations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d::AbstractVector{N}, B::Ballp)::AbstractVector{N} where {N<:AbstractFloat}\n\nReturn the support vector of a Ballp in a given direction.\n\nInput\n\nd – a direction\nB – a ball in the p-norm\n\nOutput\n\nThe support vector in the given direction. If the direction has norm zero, the center of the ball is returned.\n\nAlgorithm\n\nThe support vector of the unit ball in the p-norm along direction d is:\n\n_mathcalB_p^n(0 1)(d) = dfractildevtildev_q\n\nwhere tildev_i = fracd_i^qd_i if d_i  0 and tildev_i = 0 otherwise, for all i=1n, and q is the conjugate number of p. By the affine transformation x = rtildex + c, one obtains that the support vector of mathcalB_p^n(c r) is\n\n_mathcalB_p^n(c r)(d) = dfracvv_q\n\nwhere v_i = c_i + rfracd_i^qd_i if d_i  0 and v_i = 0 otherwise, for all i = 1  n.\n\n\n\n"
},

{
    "location": "lib/representations.html#Base.:∈-Tuple{AbstractArray{Float64,1},LazySets.Ballp{Float64}}",
    "page": "Common Set Representations",
    "title": "Base.:∈",
    "category": "Method",
    "text": "∈(x::AbstractVector{N}, B::Ballp{N})::Bool where {N<:Real}\n\nCheck whether a given point is contained in a ball in the p-norm.\n\nInput\n\nx – point/vector\nB – ball in the p-norm\n\nOutput\n\ntrue iff x  B.\n\nNotes\n\nThis implementation is worst-case optimized, i.e., it is optimistic and first computes (see below) the whole sum before comparing to the radius. In applications where the point is typically far away from the ball, a fail-fast implementation with interleaved comparisons could be more efficient.\n\nAlgorithm\n\nLet B be an n-dimensional ball in the p-norm with radius r and let c_i and x_i be the ball's center and the vector x in dimension i, respectively. Then x  B iff left( _i=1^n c_i - x_i^p right)^1p  r.\n\nExamples\n\njulia> B = Ballp(1.5, [1., 1.], 1.)\nLazySets.Ballp{Float64}(1.5, [1.0, 1.0], 1.0)\njulia> ∈([.5, -.5], B)\nfalse\njulia> ∈([.5, 1.5], B)\ntrue\n\n\n\n"
},

{
    "location": "lib/representations.html#p-norm-ball-1",
    "page": "Common Set Representations",
    "title": "p-norm ball",
    "category": "section",
    "text": "Ballp\ndim(::Ballp)\nσ(::AbstractVector{Float64}, ::Ballp)\n∈(::AbstractVector{Float64}, ::Ballp{Float64})"
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
    "text": "HPolygon{N<:Real} <: LazySet\n\nType that represents a convex polygon in constraint representation whose edges are sorted in counter-clockwise fashion with respect to their normal directions.\n\nFields\n\nconstraints_list – list of linear constraints, sorted by the angle\n\nNotes\n\nThe default constructor assumes that the given list of edges is sorted. It does not perform any sorting. Use addconstraint! to iteratively add the edges in a sorted way.\n\nHPolygon(constraints_list::Vector{LinearConstraint{<:Real}}) – default constructor\nHPolygon() – constructor with no constraints\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.addconstraint!-Tuple{LazySets.HPolygon{Float64},LazySets.LinearConstraint{Float64}}",
    "page": "Common Set Representations",
    "title": "LazySets.addconstraint!",
    "category": "Method",
    "text": "addconstraint!(P::HPolygon{N}, constraint::LinearConstraint{N})::Void where {N<:Real}\n\nAdd a linear constraint to a polygon in constraint representation, keeping the constraints sorted by their normal directions.\n\nInput\n\nP          – polygon\nconstraint – linear constraint to add\n\nOutput\n\nNothing.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.dim-Tuple{LazySets.HPolygon}",
    "page": "Common Set Representations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(P::HPolygon)::Int\n\nReturn the dimension of a polygon.\n\nInput\n\nP – polygon in constraint representation\n\nOutput\n\nThe ambient dimension of the polygon.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.HPolygon}",
    "page": "Common Set Representations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d::AbstractVector{<:Real}, P::HPolygon{N})::Vector{N} where {N<:Real}\n\nReturn the support vector of a polygon in a given direction.\n\nInput\n\nd – direction\nP – polygon in constraint representation\n\nOutput\n\nThe support vector in the given direction. The result is always one of the vertices; in particular, if the direction has norm zero, any vertex is returned.\n\nAlgorithm\n\nComparison of directions is performed using polar angles; see the overload of <= for two-dimensional vectors.\n\n\n\n"
},

{
    "location": "lib/representations.html#Base.:∈-Tuple{AbstractArray{Float64,1},LazySets.HPolygon{Float64}}",
    "page": "Common Set Representations",
    "title": "Base.:∈",
    "category": "Method",
    "text": "∈(x::AbstractVector{N}, P::HPolygon{N})::Bool where {N<:Real}\n\nCheck whether a given 2D point is contained in a polygon in constraint representation.\n\nInput\n\nx – two-dimensional point/vector\nP – polygon in constraint representation\n\nOutput\n\ntrue iff x  P.\n\nAlgorithm\n\nThis implementation checks if the point lies on the outside of each edge.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.tovrep-Tuple{LazySets.HPolygon}",
    "page": "Common Set Representations",
    "title": "LazySets.tovrep",
    "category": "Method",
    "text": "tovrep(P::HPolygon)::VPolygon\n\nBuild a vertex representation of the given polygon.\n\nInput\n\nP – polygon in constraint representation\n\nOutput\n\nThe same polygon but in vertex representation, a VPolygon.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.vertices_list-Tuple{LazySets.HPolygon}",
    "page": "Common Set Representations",
    "title": "LazySets.vertices_list",
    "category": "Method",
    "text": "vertices_list(P::HPolygon{N})::Vector{Vector{N}} where {N<:Real}\n\nReturn the list of vertices of a polygon in constraint representation.\n\nInput\n\nP – polygon in constraint representation\n\nOutput\n\nList of vertices.\n\n\n\n"
},

{
    "location": "lib/representations.html#Constraint-representation-1",
    "page": "Common Set Representations",
    "title": "Constraint representation",
    "category": "section",
    "text": "HPolygon\naddconstraint!(::HPolygon{Float64}, ::LinearConstraint{Float64})\ndim(::HPolygon)\nσ(::AbstractVector{Float64}, ::HPolygon)\n∈(::AbstractVector{Float64}, ::HPolygon{Float64})\ntovrep(::HPolygon)\nvertices_list(::HPolygon)"
},

{
    "location": "lib/representations.html#LazySets.HPolygonOpt",
    "page": "Common Set Representations",
    "title": "LazySets.HPolygonOpt",
    "category": "Type",
    "text": "HPolygonOpt{N<:Real} <: LazySet\n\nType that represents a convex polygon in constraint representation whose edges are sorted in counter-clockwise fashion with respect to their normal directions. This is a refined version of HPolygon.\n\nFields\n\nconstraints_list – list of linear constraints\nind – index in the list of constraints to begin the search to evaluate the          support function\n\nNotes\n\nThis structure is optimized to evaluate the support function/vector with a large sequence of directions that are close to each other. The strategy is to have an index that can be used to warm-start the search for optimal values in the support vector computation.\n\nThe default constructor assumes that the given list of edges is sorted. It does not perform any sorting. Use addconstraint! to iteratively add the edges in a sorted way.\n\nHPolygonOpt(constraints_list::Vector{LinearConstraint{<:Real}}, ind::Int) – default constructor\nHPolygonOpt(constraints_list::Vector{LinearConstraint{<:Real}}) – constructor without index\nHPolygonOpt(H::HPolygon{<:Real}) – constructor from an HPolygon\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.addconstraint!-Tuple{LazySets.HPolygonOpt{Float64},LazySets.LinearConstraint{Float64}}",
    "page": "Common Set Representations",
    "title": "LazySets.addconstraint!",
    "category": "Method",
    "text": "addconstraint!(P::HPolygonOpt{N}, constraint::LinearConstraint{N})::Void where {N<:Real}\n\nAdd a linear constraint to an optimized polygon in constraint representation, keeping the constraints sorted by their normal directions.\n\nInput\n\nP          – optimized polygon\nconstraint – linear constraint to add\n\nOutput\n\nNothing.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.dim-Tuple{LazySets.HPolygonOpt}",
    "page": "Common Set Representations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(P::HPolygonOpt)::Int\n\nReturn the dimension of an optimized polygon.\n\nInput\n\nP – optimized polygon in constraint representation\n\nOutput\n\nThe ambient dimension of the optimized polygon.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.HPolygonOpt{Float64}}",
    "page": "Common Set Representations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d::AbstractVector{<:Real}, P::HPolygonOpt{N})::Vector{N} where {N<:Real}\n\nReturn the support vector of an optimized polygon in a given direction.\n\nInput\n\nd – direction\nP – optimized polygon in constraint representation\n\nOutput\n\nThe support vector in the given direction. The result is always one of the vertices; in particular, if the direction has norm zero, any vertex is returned.\n\nAlgorithm\n\nComparison of directions is performed using polar angles; see the overload of <= for two-dimensional vectors.\n\n\n\n"
},

{
    "location": "lib/representations.html#Base.:∈-Tuple{AbstractArray{Float64,1},LazySets.HPolygonOpt{Float64}}",
    "page": "Common Set Representations",
    "title": "Base.:∈",
    "category": "Method",
    "text": "∈(x::AbstractVector{N}, P::HPolygonOpt{N})::Bool where {N<:Real}\n\nCheck whether a given 2D point is contained in an optimized polygon in constraint representation.\n\nInput\n\nx – two-dimensional point/vector\nP – optimized polygon in constraint representation\n\nOutput\n\ntrue iff x  P.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.tovrep-Tuple{LazySets.HPolygonOpt}",
    "page": "Common Set Representations",
    "title": "LazySets.tovrep",
    "category": "Method",
    "text": "tovrep(P::HPolygonOpt)::VPolygon\n\nBuild a vertex representation of the given optimized polygon.\n\nInput\n\nP – optimized polygon in constraint representation\n\nOutput\n\nThe same polygon but in vertex representation, a VPolygon.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.vertices_list-Tuple{LazySets.HPolygonOpt}",
    "page": "Common Set Representations",
    "title": "LazySets.vertices_list",
    "category": "Method",
    "text": "vertices_list(P::HPolygonOpt{N})::Vector{Vector{N}} where {N<:Real}\n\nReturn the list of vertices of an optimized polygon in constraint representation.\n\nInput\n\nP – an optimized polygon in constraint representation\n\nOutput\n\nList of vertices.\n\n\n\n"
},

{
    "location": "lib/representations.html#Optimized-constraint-representation-1",
    "page": "Common Set Representations",
    "title": "Optimized constraint representation",
    "category": "section",
    "text": "HPolygonOpt\naddconstraint!(::HPolygonOpt{Float64}, ::LinearConstraint{Float64})\ndim(::HPolygonOpt)\nσ(::AbstractVector{Float64}, ::HPolygonOpt{Float64})\n∈(::AbstractVector{Float64}, ::HPolygonOpt{Float64})\ntovrep(::HPolygonOpt)\nvertices_list(::HPolygonOpt)"
},

{
    "location": "lib/representations.html#LazySets.VPolygon",
    "page": "Common Set Representations",
    "title": "LazySets.VPolygon",
    "category": "Type",
    "text": "VPolygon{N<:Real} <: LazySet\n\nType that represents a polygon by its vertices.\n\nFields\n\nvertices_list – the list of vertices\n\nNotes\n\nThe constructor of VPolygon runs a convex hull algorithm, and the given vertices are sorted in counter-clockwise fashion. The constructor flag apply_convex_hull can be used to skip the computation of the convex hull.\n\nVPolygon(vertices_list::Vector{Vector{N}};           apply_convex_hull::Bool=true,           algorithm::String=\"monotone_chain\")\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.dim-Tuple{LazySets.VPolygon}",
    "page": "Common Set Representations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(P::VPolygon)::Int\n\nReturn the dimension of a polygon in vertex representation.\n\nInput\n\nP – polygon in vertex representation\n\nOutput\n\nThe ambient dimension of the polygon.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.VPolygon}",
    "page": "Common Set Representations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d::AbstractVector{<:Real}, P::VPolygon{N})::Vector{N} where {N<:Real}\n\nReturn the support vector of a polygon in a given direction.\n\nInput\n\nd – direction\nP – polygon in vertex representation\n\nOutput\n\nThe support vector in the given direction. If the direction has norm zero, the first vertex is returned.\n\nAlgorithm\n\nThis implementation performs a brute-force search, comparing the projection of each vector along the given direction. It runs in O(n) where n is the number of vertices.\n\nNotes\n\nFor arbitrary points without structure this is the best one can do. However, a more efficient approach can be used if the vertices of the polygon have been sorted in counter-clockwise fashion. In that case a binary search algorithm can be used that runs in O(log n). See issue #40.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.vertices_list-Tuple{LazySets.VPolygon}",
    "page": "Common Set Representations",
    "title": "LazySets.vertices_list",
    "category": "Method",
    "text": "vertices_list(P::VPolygon{N})::Vector{Vector{N}} where {N<:Real}\n\nReturn the list of vertices of a convex polygon in vertex representation.\n\nInput\n\nP – a polygon vertex representation\n\nOutput\n\nList of vertices.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.singleton_list-Tuple{LazySets.VPolygon}",
    "page": "Common Set Representations",
    "title": "LazySets.singleton_list",
    "category": "Method",
    "text": "singleton_list(P::VPolygon{N})::Vector{Singleton{N}} where {N<:Real}\n\nReturn the vertices of a convex polygon in vertex representation as a list of singletons.\n\nInput\n\nP – a polygon vertex representation\n\nOutput\n\nList containing a singleton for each vertex.\n\n\n\n"
},

{
    "location": "lib/representations.html#Base.:∈-Tuple{AbstractArray{Float64,1},LazySets.VPolygon{Float64}}",
    "page": "Common Set Representations",
    "title": "Base.:∈",
    "category": "Method",
    "text": "∈(x::AbstractVector{N}, P::VPolygon{N})::Bool where {N<:Real}\n\nCheck whether a given point is contained in a polygon in vertex representation.\n\nInput\n\nx – point/vector\nP – polygon in vertex representation\n\nOutput\n\ntrue iff x  P.\n\nAlgorithm\n\nThis implementation exploits that the polygon's vertices are sorted in counter-clockwise fashion. Under this assumption we can just check if the vertex lies on the left of each edge, using the dot product.\n\nExamples\n\njulia> P = VPolygon([[2.0, 3.0], [3.0, 1.0], [5.0, 1.0], [4.0, 5.0]];\n                    apply_convex_hull=false);\n\njulia> ∈([4.5, 3.1], P)\nfalse\njulia> ∈([4.5, 3.0], P)\ntrue\njulia> ∈([4.4, 3.4], P)  #  point lies on the edge -> floating point error\nfalse\njulia> P = VPolygon([[2//1, 3//1], [3//1, 1//1], [5//1, 1//1], [4//1, 5//1]];\n                     apply_convex_hull=false);\n\njulia> ∈([44//10, 34//10], P)  #  with rational numbers the answer is correct\ntrue\n\n\n\n"
},

{
    "location": "lib/representations.html#Vertex-representation-1",
    "page": "Common Set Representations",
    "title": "Vertex representation",
    "category": "section",
    "text": "VPolygon\ndim(::VPolygon)\nσ(::AbstractVector{Float64}, ::VPolygon)\nvertices_list(::VPolygon)\nsingleton_list(::VPolygon)\n∈(::AbstractVector{Float64}, ::VPolygon{Float64})"
},

{
    "location": "lib/representations.html#LazySets.LinearConstraint",
    "page": "Common Set Representations",
    "title": "LazySets.LinearConstraint",
    "category": "Type",
    "text": "LinearConstraint{N<:Real}\n\nType that represents a linear constraint (a half-space) of the form ax  b.\n\nFields\n\na – normal direction\nb – constraint\n\nExamples\n\nThe set y  0 in the plane:\n\njulia> LinearConstraint([0, -1.], 0.)\nLazySets.LinearConstraint{Float64}([0.0, -1.0], 0.0)\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.Line",
    "page": "Common Set Representations",
    "title": "LazySets.Line",
    "category": "Type",
    "text": "Line{N<:Real}\n\nType that represents a line in 2D of the form ax = b.\n\nFields\n\na – normal direction\nb – constraint\n\nExamples\n\nThe line y = -x + 1:\n\njulia> Line([1., 1.], 1.)\nLazySets.Line{Float64}([1.0, 1.0], 1.0)\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.intersection-Tuple{LazySets.Line{Float64},LazySets.Line{Float64}}",
    "page": "Common Set Representations",
    "title": "LazySets.intersection",
    "category": "Method",
    "text": "intersection(L1::Line{N}, L2::Line{N})::Vector{N} where {N<:Real}\n\nReturn the intersection of two 2D lines.\n\nInput\n\nL1 – first line\nL2 – second line\n\nOutput\n\nIf the lines are parallel or identical, the result is an empty vector. Otherwise the result is the only intersection point.\n\nExamples\n\nThe line y = -x + 1 intersected with the line y = x:\n\njulia> intersection(Line([-1., 1.], 0.), Line([1., 1.], 1.))\n2-element Array{Float64,1}:\n 0.5\n 0.5\njulia> intersection(Line([1., 1.], 1.), Line([1., 1.], 1.))\n0-element Array{Float64,1}\n\n\n\n\n"
},

{
    "location": "lib/representations.html#Lines-and-linear-constraints-1",
    "page": "Common Set Representations",
    "title": "Lines and linear constraints",
    "category": "section",
    "text": "LinearConstraint\nLine\nintersection(::Line{Float64}, L2::Line{Float64})"
},

{
    "location": "lib/representations.html#LazySets.Hyperrectangle",
    "page": "Common Set Representations",
    "title": "LazySets.Hyperrectangle",
    "category": "Type",
    "text": "Hyperrectangle{N<:Real} <: LazySet\n\nType that represents a hyperrectangle.\n\nA hyperrectangle is the Cartesian product of one-dimensional intervals.\n\nFields\n\ncenter – center of the hyperrectangle as a real vector\nradius – radius of the ball as a real vector, i.e., half of its width along             each coordinate direction\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.Hyperrectangle-Tuple{}",
    "page": "Common Set Representations",
    "title": "LazySets.Hyperrectangle",
    "category": "Method",
    "text": "Hyperrectangle(;kwargs...)\n\nConstruct a hyperrectangle from keyword arguments.\n\nInput\n\nkwargs – keyword arguments; two combinations are allowed:\ncenter, radius – vectors\nhigh, low      – vectors (if both center and radius are also                       defined, those are chosen instead)\n\nOutput\n\nA hyperrectangle.\n\nExamples\n\nThe following three constructions are equivalent:\n\njulia> c = ones(2);\n\njulia> r = [0.1, 0.2];\n\njulia> l = [0.9, 0.8];\n\njulia> h = [1.1, 1.2];\n\njulia> H1 = Hyperrectangle(c, r)\nLazySets.Hyperrectangle{Float64}([1.0, 1.0], [0.1, 0.2])\njulia> H2 = Hyperrectangle(center=c, radius=r)\nLazySets.Hyperrectangle{Float64}([1.0, 1.0], [0.1, 0.2])\njulia> H3 = Hyperrectangle(low=l, high=h)\nLazySets.Hyperrectangle{Float64}([1.0, 1.0], [0.1, 0.2])\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.dim-Tuple{LazySets.Hyperrectangle}",
    "page": "Common Set Representations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(H::Hyperrectangle)::Int\n\nReturn the dimension of a hyperrectangle.\n\nInput\n\nH – hyperrectangle\n\nOutput\n\nThe ambient dimension of the hyperrectangle.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.Hyperrectangle}",
    "page": "Common Set Representations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d::AbstractVector{<:Real}, H::Hyperrectangle)::AbstractVector{<:Real}\n\nReturn the support vector of a hyperrectangle in a given direction.\n\nInput\n\nd – direction\nH – hyperrectangle\n\nOutput\n\nThe support vector in the given direction. If the direction has norm zero, the vertex with biggest values is returned.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.vertices_list-Tuple{LazySets.Hyperrectangle}",
    "page": "Common Set Representations",
    "title": "LazySets.vertices_list",
    "category": "Method",
    "text": "vertices_list(H::Hyperrectangle{N})::Vector{Vector{N}} where {N<:Real}\n\nReturn the vertices of a hyperrectangle.\n\nInput\n\nH – hyperrectangle\n\nOutput\n\nA list of vertices.\n\nNotes\n\nFor high dimensions, it is preferable to develop a vertex_iterator approach.\n\n\n\n"
},

{
    "location": "lib/representations.html#Base.LinAlg.norm",
    "page": "Common Set Representations",
    "title": "Base.LinAlg.norm",
    "category": "Function",
    "text": "norm(H::Hyperrectangle, [p]::Real=Inf)::Real\n\nReturn the norm of a hyperrectangle.\n\nInput\n\nH – hyperrectangle\np – (optional, default: Inf) norm\n\nOutput\n\nA real number representing the norm.\n\nNotes\n\nThe norm of a hyperrectangle is defined as the norm of the enclosing ball, of the given p-norm, of minimal volume.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.radius",
    "page": "Common Set Representations",
    "title": "LazySets.radius",
    "category": "Function",
    "text": "radius(H::Hyperrectangle, [p]::Real=Inf)::Real\n\nReturn the radius of a hyperrectangle.\n\nInput\n\nH – hyperrectangle\np – (optional, default: Inf) norm\n\nOutput\n\nA real number representing the radius.\n\nNotes\n\nThe radius is defined as the radius of the enclosing ball of the given p-norm of minimal volume with the same center.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.diameter",
    "page": "Common Set Representations",
    "title": "LazySets.diameter",
    "category": "Function",
    "text": "diameter(H::Hyperrectangle, [p]::Real=Inf)::Real\n\nReturn the diameter of a hyperrectangle.\n\nInput\n\nH – hyperrectangle\np – (optional, default: Inf) norm\n\nOutput\n\nA real number representing the diameter.\n\nNotes\n\nThe diameter is defined as the maximum distance in the given p-norm between any two elements of the set. Equivalently, it is the diameter of the enclosing ball of the given p-norm of minimal volume with the same center.\n\n\n\n"
},

{
    "location": "lib/representations.html#Base.:∈-Tuple{AbstractArray{Float64,1},LazySets.Hyperrectangle{Float64}}",
    "page": "Common Set Representations",
    "title": "Base.:∈",
    "category": "Method",
    "text": "∈(x::AbstractVector{N}, H::Hyperrectangle{N})::Bool where {N<:Real}\n\nCheck whether a given point is contained in a hyperrectangle.\n\nInput\n\nx – point/vector\nH – hyperrectangle\n\nOutput\n\ntrue iff x  H.\n\nAlgorithm\n\nLet H be an n-dimensional hyperrectangle, c_i and r_i be the ball's center and radius and x_i be the vector x in dimension i, respectively. Then x  H iff c_i - x_i  r_i for all i=1n.\n\nExamples\n\njulia> H = Hyperrectangle([1.0, 1.0], [2.0, 3.0]);\n\njulia> ∈([-1.1, 4.1], H)\nfalse\njulia> ∈([-1.0, 4.0], H)\ntrue\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.high-Tuple{LazySets.Hyperrectangle}",
    "page": "Common Set Representations",
    "title": "LazySets.high",
    "category": "Method",
    "text": "high(H::Hyperrectangle{N})::Vector{N} where {N<:Real}\n\nReturn the higher coordinates of a hyperrectangle.\n\nInput\n\nH – hyperrectangle\n\nOutput\n\nA vector with the higher coordinates of the hyperrectangle, one entry per dimension.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.low-Tuple{LazySets.Hyperrectangle}",
    "page": "Common Set Representations",
    "title": "LazySets.low",
    "category": "Method",
    "text": "low(H::Hyperrectangle{N})::Vector{N} where {N<:Real}\n\nReturn the lower coordinates of a hyperrectangle.\n\nInput\n\nH – hyperrectangle\n\nOutput\n\nA vector with the lower coordinates of the hyperrectangle, one entry per dimension.\n\n\n\n"
},

{
    "location": "lib/representations.html#Hyperrectangles-1",
    "page": "Common Set Representations",
    "title": "Hyperrectangles",
    "category": "section",
    "text": "Hyperrectangle\nHyperrectangle(;kwargs...)\ndim(::Hyperrectangle)\nσ(::AbstractVector{Float64}, ::Hyperrectangle)\nvertices_list(::Hyperrectangle)\nnorm(::Hyperrectangle, ::Real=Inf)\nradius(::Hyperrectangle, ::Real=Inf)\ndiameter(::Hyperrectangle, ::Real=Inf)\n∈(::AbstractVector{Float64}, ::Hyperrectangle{Float64})\nhigh(::Hyperrectangle)\nlow(::Hyperrectangle)"
},

{
    "location": "lib/representations.html#LazySets.EmptySet",
    "page": "Common Set Representations",
    "title": "LazySets.EmptySet",
    "category": "Type",
    "text": "EmptySet <: LazySet\n\nType that represents the empty set, i.e., the set with no elements.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.dim-Tuple{LazySets.EmptySet}",
    "page": "Common Set Representations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(S::EmptySet)\n\nReturn the dimension of the empty set, which is -1 by convention.\n\nInput\n\nS – an empty set\n\nOutput\n\n-1 by convention.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.EmptySet}",
    "page": "Common Set Representations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d, ∅)\n\nReturn the support vector of an empty set.\n\nInput\n\n∅ – an empty set\n\nOutput\n\nAn error.\n\n\n\n"
},

{
    "location": "lib/representations.html#Base.:∈-Tuple{AbstractArray{Float64,1},LazySets.EmptySet}",
    "page": "Common Set Representations",
    "title": "Base.:∈",
    "category": "Method",
    "text": "∈(x::AbstractVector, ∅::EmptySet)::Bool\n\nCheck whether a given point is contained in an empty set.\n\nInput\n\nx – point/vector\n∅ – empty set\n\nOutput\n\nThe output is always false.\n\nExamples\n\njulia> ∈([1.0, 0.0], ∅)\nfalse\n\n\n\n"
},

{
    "location": "lib/representations.html#EmptySet-1",
    "page": "Common Set Representations",
    "title": "EmptySet",
    "category": "section",
    "text": "EmptySet\ndim(::EmptySet)\nσ(::AbstractVector{Float64}, ::EmptySet)\n∈(::AbstractVector{Float64}, ::EmptySet)"
},

{
    "location": "lib/representations.html#LazySets.ZeroSet",
    "page": "Common Set Representations",
    "title": "LazySets.ZeroSet",
    "category": "Type",
    "text": "ZeroSet <: LazySet\n\nType that represents the zero set, i.e., the set that only contains the origin.\n\nFields\n\ndim – the ambient dimension of this zero set\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.dim-Tuple{LazySets.ZeroSet}",
    "page": "Common Set Representations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(Z::ZeroSet)::Int\n\nReturn the ambient dimension of this zero set.\n\nInput\n\nZ – a zero set, i.e., a set that only contains the origin\n\nOutput\n\nThe ambient dimension of the zero set.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.ZeroSet}",
    "page": "Common Set Representations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d, Z)\n\nReturn the support vector of a zero set.\n\nInput\n\nZ – a zero set, i.e., a set that only contains the origin\n\nOutput\n\nThe returned value is the origin since it is the only point that belongs to this set.\n\n\n\n"
},

{
    "location": "lib/representations.html#Base.:∈-Tuple{AbstractArray{Float64,1},LazySets.ZeroSet}",
    "page": "Common Set Representations",
    "title": "Base.:∈",
    "category": "Method",
    "text": "∈(x::AbstractVector, Z::ZeroSet)::Bool\n\nCheck whether a given point is contained in a zero set.\n\nInput\n\nx – point/vector\nZ – zero set\n\nOutput\n\ntrue iff x  Z.\n\nExamples\n\njulia> Z = ZeroSet(2);\n\njulia> ∈([1.0, 0.0], Z)\nfalse\njulia> ∈([0.0, 0.0], Z)\ntrue\n\n\n\n"
},

{
    "location": "lib/representations.html#ZeroSet-1",
    "page": "Common Set Representations",
    "title": "ZeroSet",
    "category": "section",
    "text": "ZeroSet\ndim(::ZeroSet)\nσ(::AbstractVector{Float64}, ::ZeroSet)\n∈(::AbstractVector{Float64}, ::ZeroSet)"
},

{
    "location": "lib/representations.html#LazySets.Singleton",
    "page": "Common Set Representations",
    "title": "LazySets.Singleton",
    "category": "Type",
    "text": "Singleton{N<:Real} <: LazySet\n\nType that represents a singleton, that is, a set with a unique element.\n\nFields\n\nelement – the only element of the set\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.dim-Tuple{LazySets.Singleton}",
    "page": "Common Set Representations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(S::Singleton)::Int\n\nReturn the dimension of a singleton.\n\nInput\n\nS – singleton\n\nOutput\n\nThe ambient dimension of the singleton.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.Singleton}",
    "page": "Common Set Representations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d::AbstractVector{<:Real}, S::LazySets.Singleton{N})::Vector{N} where {N<:Real}\n\nReturn the support vector of a singleton.\n\nInput\n\nd – direction\nB – singleton\n\nOutput\n\nThe support vector, which is the singleton's vector itself, irrespective of the given direction.\n\n\n\n"
},

{
    "location": "lib/representations.html#Base.:∈-Tuple{AbstractArray{Float64,1},LazySets.Singleton{Float64}}",
    "page": "Common Set Representations",
    "title": "Base.:∈",
    "category": "Method",
    "text": "∈(x::AbstractVector{N}, S::Singleton{N})::Bool where {N<:Real}\n\nCheck whether a given point is contained in a singleton.\n\nInput\n\nx – point/vector\nS – singleton\n\nOutput\n\ntrue iff x  S.\n\nNotes\n\nThis implementation performs an exact comparison, which may be insufficient with floating point computations.\n\nExamples\n\njulia> S = Singleton([1., 1.]);\n\njulia> ∈([0.9, 1.1], S)\nfalse\njulia> ∈([1.0, 1.0], S)\ntrue\n\n\n\n"
},

{
    "location": "lib/representations.html#Base.:⊆-Tuple{LazySets.Singleton,LazySets.LazySet}",
    "page": "Common Set Representations",
    "title": "Base.:⊆",
    "category": "Method",
    "text": "⊆(S::Singleton, set::LazySet)::Bool\n\nCheck whether a given singleton is contained in a convex set.\n\nInput\n\nS   – singleton\nset – convex set\n\nOutput\n\ntrue iff S  textset.\n\n\n\n"
},

{
    "location": "lib/representations.html#Singletons-1",
    "page": "Common Set Representations",
    "title": "Singletons",
    "category": "section",
    "text": "Singleton\ndim(::Singleton)\nσ(::AbstractVector{Float64}, ::Singleton)\n∈(::AbstractVector{Float64}, ::Singleton{Float64})\n⊆(::Singleton, ::LazySet)"
},

{
    "location": "lib/representations.html#LazySets.Zonotope",
    "page": "Common Set Representations",
    "title": "LazySets.Zonotope",
    "category": "Type",
    "text": "Zonotope{N<:Real} <: LazySet\n\nType that represents a zonotope.\n\nFields\n\ncenter     – center of the zonotope\ngenerators – matrix; each column is a generator of the zonotope\n\nNotes\n\nMathematically, a zonotope is defined as the set\n\nZ = left c + _i=1^p _i g_i _i in -1 1  i = 1 p right\n\nwhere c in mathbbR^n is its center and g_i_i=1^p, g_i in mathbbR^n, is the set of generators. This characterization defines a zonotope as the finite Minkowski sum of line elements. Zonotopes can be equivalently described as the image of a unit infinity-norm ball in mathbbR^n by an affine transformation.\n\nZonotope(center::AbstractVector{N},           generators::AbstractMatrix{N}) where {N<:Real}\nZonotope(center::AbstractVector{N},           generators_list::AbstractVector{T}) where {N<:Real, T<:AbstractVector{N}}\n\nExamples\n\nA two-dimensional zonotope with given center and set of generators:\n\njulia> Z = Zonotope([1.0, 0.0], 0.1*eye(2))\nLazySets.Zonotope{Float64}([1.0, 0.0], [0.1 0.0; 0.0 0.1])\njulia> dim(Z)\n2\n\nCompute its vertices:\n\njulia> vertices_list(Z)\n4-element Array{Array{Float64,1},1}:\n [0.9, -0.1]\n [1.1, -0.1]\n [1.1, 0.1]\n [0.9, 0.1]\n\nEvaluate the support vector in a given direction:\n\njulia> σ([1., 1.], Z)\n2-element Array{Float64,1}:\n 1.1\n 0.1\n\nAlternative constructor: A zonotope in two dimensions with three generators:\n\njulia> Z = Zonotope(ones(2), [[1., 0.], [0., 1.], [1., 1.]])\nLazySets.Zonotope{Float64}([1.0, 1.0], [1.0 0.0 1.0; 0.0 1.0 1.0])\njulia> Z.generators\n2×3 Array{Float64,2}:\n 1.0  0.0  1.0\n 0.0  1.0  1.0\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.dim-Tuple{LazySets.Zonotope}",
    "page": "Common Set Representations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(Z::Zonotope)::Int\n\nReturn the dimension of a zonotope.\n\nInput\n\nZ – zonotope\n\nOutput\n\nThe ambient dimension of the zonotope.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.Zonotope}",
    "page": "Common Set Representations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d::AbstractVector{<:Real}, Z::Zonotope)::AbstractVector{<:Real}\n\nReturn the support vector of a zonotope in a given direction.\n\nInput\n\nd – direction\nZ – zonotope\n\nOutput\n\nSupport vector in the given direction. If the direction has norm zero, the vertex with _i = 1    i = 1 p is returned.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.vertices_list-Tuple{LazySets.Zonotope{Float64}}",
    "page": "Common Set Representations",
    "title": "LazySets.vertices_list",
    "category": "Method",
    "text": "vertices_list(Z::Zonotope{N})::Vector{Vector{N}} where {N<:Real}\n\nReturn the vertices of a zonotope.\n\nInput\n\nZ – zonotope\n\nOutput\n\nList of vertices.\n\nNotes\n\nThis implementation computes a convex hull.\n\nFor high dimensions, it would be preferable to develop a vertex_iterator approach.\n\n\n\n"
},

{
    "location": "lib/representations.html#LazySets.order-Tuple{LazySets.Zonotope}",
    "page": "Common Set Representations",
    "title": "LazySets.order",
    "category": "Method",
    "text": "order(Z::Zonotope)::Rational\n\nReturn the order of a zonotope.\n\nInput\n\nZ – zonotope\n\nOutput\n\nA rational number representing the order of the zonotope.\n\nNotes\n\nThe order of a zonotope is defined as the quotient of its number of generators and its dimension.\n\n\n\n"
},

{
    "location": "lib/representations.html#Base.:∈-Tuple{AbstractArray{Float64,1},LazySets.Zonotope{Float64}}",
    "page": "Common Set Representations",
    "title": "Base.:∈",
    "category": "Method",
    "text": "∈(x::AbstractVector{N}, Z::Zonotope{N})::Bool where {N<:Real}\n\nCheck whether a given point is contained in a zonotope.\n\nInput\n\nx – point/vector\nZ – zonotope\n\nOutput\n\ntrue iff x  Z.\n\nAlgorithm\n\nThis implementation poses the problem as a linear equality system and solves it using Base.:. A zonotope centered in the origin with generators g_i contains a point x iff x = _i=1^p _i g_i for some _i in -1 1  i = 1 p. Thus, we first ask for a solution and then check if it is in this Cartesian product of intervals.\n\nOther algorithms exist which test the feasibility of an LP.\n\nExamples\n\njulia> Z = Zonotope([1.0, 0.0], 0.1*eye(2));\n\njulia> ∈([1.0, 0.2], Z)\nfalse\njulia> ∈([1.0, 0.1], Z)\ntrue\n\n\n\n"
},

{
    "location": "lib/representations.html#Zonotopes-1",
    "page": "Common Set Representations",
    "title": "Zonotopes",
    "category": "section",
    "text": "Zonotope\ndim(::Zonotope)\nσ(d::AbstractVector{Float64}, Z::Zonotope)\nvertices_list(::Zonotope{Float64})\norder(::Zonotope)\n∈(::AbstractVector{Float64}, ::Zonotope{Float64})"
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
    "text": "This section of the manual describes the basic symbolic types describing operations between sets.Pages = [\"operations.md\"]\nDepth = 3CurrentModule = LazySets\nDocTestSetup = quote\n    using LazySets\nend"
},

{
    "location": "lib/operations.html#Minkowski-Sum-1",
    "page": "Common Set Operations",
    "title": "Minkowski Sum",
    "category": "section",
    "text": ""
},

{
    "location": "lib/operations.html#Binary-Minkowski-Sum-1",
    "page": "Common Set Operations",
    "title": "Binary Minkowski Sum",
    "category": "section",
    "text": "MinkowskiSum\ndim(::MinkowskiSum)\nσ(::AbstractVector{Float64}, ::MinkowskiSum)\nBase.:+(::LazySet, ::LazySet)\n⊕"
},

{
    "location": "lib/operations.html#LazySets.MinkowskiSumArray",
    "page": "Common Set Operations",
    "title": "LazySets.MinkowskiSumArray",
    "category": "Type",
    "text": "MinkowskiSumArray{T<:LazySet} <: LazySet\n\nType that represents the Minkowski sum of a finite number of convex sets.\n\nFields\n\nsfarray – array of convex sets\n\nNotes\n\nThis type assumes that the dimensions of all elements match.\n\nMinkowskiSumArray(sfarray::Vector{<:LazySet}) – default constructor\nMinkowskiSumArray() – constructor for an empty sum\nMinkowskiSumArray(n::Int) – constructor for an empty sum with size hint\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.dim-Tuple{LazySets.MinkowskiSumArray}",
    "page": "Common Set Operations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(msa::MinkowskiSumArray)::Int\n\nReturn the dimension of a Minkowski sum of a finite number of sets.\n\nInput\n\nmsa – Minkowski sum array\n\nOutput\n\nThe ambient dimension of the Minkowski sum of a finite number of sets.\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.MinkowskiSumArray}",
    "page": "Common Set Operations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d::AbstractVector{<:Real}, msa::MinkowskiSumArray)::Vector{<:Real}\n\nReturn the support vector of a Minkowski sum of a finite number of sets in a given direction.\n\nInput\n\nd   – direction\nmsa – Minkowski sum array\n\nOutput\n\nThe support vector in the given direction. If the direction has norm zero, the result depends on the summand sets.\n\n\n\n"
},

{
    "location": "lib/operations.html#Base.:+-Tuple{LazySets.MinkowskiSumArray,LazySets.LazySet}",
    "page": "Common Set Operations",
    "title": "Base.:+",
    "category": "Method",
    "text": "+(msa::MinkowskiSumArray, S::LazySet)::MinkowskiSumArray\n\nAdd a convex set to a Minkowski sum of a finite number of convex sets from the right.\n\nInput\n\nmsa – Minkowski sum array (is modified)\nS   – convex set\n\nOutput\n\nThe modified Minkowski sum of a finite number of convex sets.\n\n\n\n"
},

{
    "location": "lib/operations.html#Base.:+-Tuple{LazySets.LazySet,LazySets.MinkowskiSumArray}",
    "page": "Common Set Operations",
    "title": "Base.:+",
    "category": "Method",
    "text": "+(S::LazySet, msa::MinkowskiSumArray)::MinkowskiSumArray\n\nAdd a convex set to a Minkowski sum of a finite number of convex sets from the left.\n\nInput\n\nS   – convex set\nmsa – Minkowski sum array (is modified)\n\nOutput\n\nThe modified Minkowski sum of a finite number of convex sets.\n\n\n\n"
},

{
    "location": "lib/operations.html#Base.:+-Tuple{LazySets.MinkowskiSumArray,LazySets.MinkowskiSumArray}",
    "page": "Common Set Operations",
    "title": "Base.:+",
    "category": "Method",
    "text": "+(msa1::MinkowskiSumArray, msa2::MinkowskiSumArray)::MinkowskiSumArray\n\nAdd the elements of a finite Minkowski sum of convex sets to another finite Minkowski sum.\n\nInput\n\nmsa1 – first Minkowski sum array (is modified)\nmsa2 – second Minkowski sum array\n\nOutput\n\nThe modified first Minkowski sum of a finite number of convex sets.\n\n\n\n"
},

{
    "location": "lib/operations.html#Base.:+-Tuple{LazySets.MinkowskiSumArray,LazySets.ZeroSet}",
    "page": "Common Set Operations",
    "title": "Base.:+",
    "category": "Method",
    "text": "+(msa::MinkowskiSumArray, Z::ZeroSet)::MinkowskiSumArray\n\nReturns the original array because addition with an empty set is a no-op.\n\nInput\n\nmsa – Minkowski sum array\nZ  – a Zero set\n\n\n\n"
},

{
    "location": "lib/operations.html#n-ary-Minkowski-Sum-1",
    "page": "Common Set Operations",
    "title": "n-ary Minkowski Sum",
    "category": "section",
    "text": "MinkowskiSumArray\ndim(::MinkowskiSumArray)\nσ(::AbstractVector{Float64}, ::MinkowskiSumArray)\nBase.:+(::MinkowskiSumArray, ::LazySet)\nBase.:+(::LazySet, ::MinkowskiSumArray)\nBase.:+(::MinkowskiSumArray, ::MinkowskiSumArray)\nBase.:+(::MinkowskiSumArray, ::ZeroSet)"
},

{
    "location": "lib/operations.html#Cartesian-Product-1",
    "page": "Common Set Operations",
    "title": "Cartesian Product",
    "category": "section",
    "text": ""
},

{
    "location": "lib/operations.html#LazySets.CartesianProduct",
    "page": "Common Set Operations",
    "title": "LazySets.CartesianProduct",
    "category": "Type",
    "text": "CartesianProduct{S1<:LazySet,S2<:LazySet} <: LazySet\n\nType that represents a Cartesian product of two convex sets.\n\nFields\n\nX – first convex set\nY – second convex set\n\nNotes\n\nThe Cartesian product of three elements is obtained recursively. See also CartesianProductArray for an implementation of a Cartesian product of many sets without recursion, instead using an array.\n\nCartesianProduct{S1<:LazySet,S2<:LazySet}            – default constructor\nCartesianProduct(Xarr::Vector{S}) where {S<:LazySet} – constructor from an                                                           array of convex sets\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.dim-Tuple{LazySets.CartesianProduct}",
    "page": "Common Set Operations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(cp::CartesianProduct)::Int\n\nReturn the dimension of a Cartesian product.\n\nInput\n\ncp – Cartesian product\n\nOutput\n\nThe ambient dimension of the Cartesian product.\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.CartesianProduct}",
    "page": "Common Set Operations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d::AbstractVector{<:Real}, cp::CartesianProduct)::AbstractVector{<:Real}\n\nReturn the support vector of a Cartesian product.\n\nInput\n\nd  – direction\ncp – Cartesian product\n\nOutput\n\nThe support vector in the given direction. If the direction has norm zero, the result depends on the product sets.\n\n\n\n"
},

{
    "location": "lib/operations.html#Base.:*-Tuple{LazySets.LazySet,LazySets.LazySet}",
    "page": "Common Set Operations",
    "title": "Base.:*",
    "category": "Method",
    "text": "    *(X::LazySet, Y::LazySet)::CartesianProduct\n\nReturn the Cartesian product of two convex sets.\n\nInput\n\nX – convex set\nY – convex set\n\nOutput\n\nThe Cartesian product of the two convex sets.\n\n\n\n"
},

{
    "location": "lib/operations.html#Base.:∈-Tuple{AbstractArray{Float64,1},LazySets.CartesianProduct}",
    "page": "Common Set Operations",
    "title": "Base.:∈",
    "category": "Method",
    "text": "∈(x::AbstractVector{<:Real}, cp::CartesianProduct)::Bool\n\nCheck whether a given point is contained in a Cartesian product set.\n\nInput\n\nx  – point/vector\ncp – Cartesian product\n\nOutput\n\ntrue iff x  cp.\n\n\n\n"
},

{
    "location": "lib/operations.html#Binary-Cartesian-Product-1",
    "page": "Common Set Operations",
    "title": "Binary Cartesian Product",
    "category": "section",
    "text": "CartesianProduct\ndim(::CartesianProduct)\nσ(::AbstractVector{Float64}, ::CartesianProduct)\nBase.:*(::LazySet, ::LazySet)\n∈(::AbstractVector{Float64}, ::CartesianProduct)"
},

{
    "location": "lib/operations.html#LazySets.CartesianProductArray",
    "page": "Common Set Operations",
    "title": "LazySets.CartesianProductArray",
    "category": "Type",
    "text": "CartesianProductArray{S<:LazySet} <: LazySet\n\nType that represents the Cartesian product of a finite number of convex sets.\n\nFields\n\nsfarray – array of sets\n\nNotes\n\nCartesianProductArray(sfarray::Vector{<:LazySet}) – default constructor\nCartesianProductArray() – constructor for an empty Cartesian product\nCartesianProductArray(n::Int) – constructor for an empty Cartesian product with size hint\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.dim-Tuple{LazySets.CartesianProductArray}",
    "page": "Common Set Operations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(cpa::CartesianProductArray)::Int\n\nReturn the dimension of a Cartesian product of a finite number of convex sets.\n\nInput\n\ncpa – Cartesian product array\n\nOutput\n\nThe ambient dimension of the Cartesian product of a finite number of convex sets.\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.CartesianProductArray}",
    "page": "Common Set Operations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d::AbstractVector{<:Real}, cpa::CartesianProductArray)::AbstractVector{<:Real}\n\nSupport vector of a Cartesian product.\n\nInput\n\nd   – direction\ncpa – Cartesian product array\n\nOutput\n\nThe support vector in the given direction. If the direction has norm zero, the result depends on the product sets.\n\n\n\n"
},

{
    "location": "lib/operations.html#Base.:*-Tuple{LazySets.CartesianProductArray,LazySets.LazySet}",
    "page": "Common Set Operations",
    "title": "Base.:*",
    "category": "Method",
    "text": "    *(cpa::CartesianProductArray, S::LazySet)::CartesianProductArray\n\nMultiply a convex set to a Cartesian product of a finite number of convex sets from the right.\n\nInput\n\ncpa – Cartesian product array (is modified)\nS   – convex set\n\nOutput\n\nThe modified Cartesian product of a finite number of convex sets.\n\n\n\n"
},

{
    "location": "lib/operations.html#Base.:*-Tuple{LazySets.LazySet,LazySets.CartesianProductArray}",
    "page": "Common Set Operations",
    "title": "Base.:*",
    "category": "Method",
    "text": "    *(S::LazySet, cpa::CartesianProductArray)::CartesianProductArray\n\nMultiply a convex set to a Cartesian product of a finite number of convex sets from the left.\n\nInput\n\nS   – convex set\ncpa – Cartesian product array (is modified)\n\nOutput\n\nThe modified Cartesian product of a finite number of convex sets.\n\n\n\n"
},

{
    "location": "lib/operations.html#Base.:*-Tuple{LazySets.CartesianProductArray,LazySets.CartesianProductArray}",
    "page": "Common Set Operations",
    "title": "Base.:*",
    "category": "Method",
    "text": "    *(cpa1::CartesianProductArray, cpa2::CartesianProductArray)::CartesianProductArray\n\nMultiply a finite Cartesian product of convex sets to another finite Cartesian product.\n\nInput\n\ncpa1 – first Cartesian product array (is modified)\ncpa2 – second Cartesian product array\n\nOutput\n\nThe modified first Cartesian product.\n\n\n\n"
},

{
    "location": "lib/operations.html#Base.:∈-Tuple{AbstractArray{Float64,1},LazySets.CartesianProductArray}",
    "page": "Common Set Operations",
    "title": "Base.:∈",
    "category": "Method",
    "text": "∈(x::AbstractVector{<:Real}, cpa::CartesianProductArray)::Bool\n\nCheck whether a given point is contained in a Cartesian product of a finite number of sets.\n\nInput\n\nx   – point/vector\ncpa – Cartesian product array\n\nOutput\n\ntrue iff x  textcpa.\n\n\n\n"
},

{
    "location": "lib/operations.html#n-ary-Cartesian-Product-1",
    "page": "Common Set Operations",
    "title": "n-ary Cartesian Product",
    "category": "section",
    "text": "CartesianProductArray\ndim(::CartesianProductArray)\nσ(::AbstractVector{Float64}, ::CartesianProductArray)\nBase.:*(::CartesianProductArray, ::LazySet)\nBase.:*(::LazySet, ::CartesianProductArray)\nBase.:*(::CartesianProductArray, ::CartesianProductArray)\n∈(::AbstractVector{Float64}, ::CartesianProductArray)"
},

{
    "location": "lib/operations.html#Maps-1",
    "page": "Common Set Operations",
    "title": "Maps",
    "category": "section",
    "text": ""
},

{
    "location": "lib/operations.html#LazySets.LinearMap",
    "page": "Common Set Operations",
    "title": "LazySets.LinearMap",
    "category": "Type",
    "text": "LinearMap{S<:LazySet, N<:Real} <: LazySet\n\nType that represents a linear transformation MS of a convex set S.\n\nFields\n\nM  – matrix/linear map\nsf – convex set\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.dim-Tuple{LazySets.LinearMap}",
    "page": "Common Set Operations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(lm::LinearMap)::Int\n\nReturn the dimension of a linear map.\n\nInput\n\nlm – linear map\n\nOutput\n\nThe ambient dimension of the linear map.\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.LinearMap}",
    "page": "Common Set Operations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d::AbstractVector{<:Real}, lm::LinearMap)::AbstractVector{<:Real}\n\nReturn the support vector of the linear map.\n\nInput\n\nd  – direction\nlm – linear map\n\nOutput\n\nThe support vector in the given direction. If the direction has norm zero, the result depends on the wrapped set.\n\nNotes\n\nIf L = MS, where M is a matrix and S is a convex set, it follows that (d L) = M(M^T d S) for any direction d.\n\n\n\n"
},

{
    "location": "lib/operations.html#Base.:*-Tuple{AbstractArray{Float64,2},LazySets.LazySet}",
    "page": "Common Set Operations",
    "title": "Base.:*",
    "category": "Method",
    "text": "    *(M::AbstractMatrix{<:Real}, S::LazySet)\n\nReturn the linear map of a convex set.\n\nInput\n\nM – matrix/linear map\nS – convex set\n\nOutput\n\nIf the matrix is null, a ZeroSet is returned; otherwise a lazy linear map.\n\n\n\n"
},

{
    "location": "lib/operations.html#Base.:*-Tuple{Real,LazySets.LazySet}",
    "page": "Common Set Operations",
    "title": "Base.:*",
    "category": "Method",
    "text": "    *(a::Real, S::LazySet)::LinearMap\n\nReturn a linear map of a convex set by a scalar value.\n\nInput\n\na – real scalar\nS – convex set\n\nOutput\n\nThe linear map of the convex set.\n\n\n\n"
},

{
    "location": "lib/operations.html#Base.:∈-Tuple{AbstractArray{Float64,1},LazySets.LinearMap{#s1,Float64} where #s1<:LazySets.LazySet}",
    "page": "Common Set Operations",
    "title": "Base.:∈",
    "category": "Method",
    "text": "∈(x::AbstractVector{N}, lm::LinearMap{<:LazySet, N})::Bool where {N<:Real}\n\nCheck whether a given point is contained in a linear map of a convex set.\n\nInput\n\nx  – point/vector\nlm – linear map of a convex set\n\nOutput\n\ntrue iff x  lm.\n\nAlgorithm\n\nNote that x  MS iff M^-1x  S. This implementation does not explicitly invert the matrix, which is why it also works for non-square matrices.\n\nExamples\n\njulia> lm = LinearMap([2.0 0.0; 0.0 1.0], BallInf([1., 1.], 1.));\n\njulia> ∈([5.0, 1.0], lm)\nfalse\njulia> ∈([3.0, 1.0], lm)\ntrue\n\nAn example with non-square matrix:\n\njulia> B = BallInf(zeros(4), 1.);\n\njulia> M = [1. 0 0 0; 0 1 0 0]/2;\n\njulia> ∈([0.5, 0.5], M*B)\ntrue\n\n\n\n"
},

{
    "location": "lib/operations.html#Linear-Map-1",
    "page": "Common Set Operations",
    "title": "Linear Map",
    "category": "section",
    "text": "LinearMap\ndim(::LinearMap)\nσ(::AbstractVector{Float64}, ::LinearMap)\n*(::AbstractMatrix{Float64}, ::LazySet)\n*(::Real, ::LazySet)\n∈(::AbstractVector{Float64}, ::LinearMap{<:LazySet, Float64})"
},

{
    "location": "lib/operations.html#LazySets.ExponentialMap",
    "page": "Common Set Operations",
    "title": "LazySets.ExponentialMap",
    "category": "Type",
    "text": "ExponentialMap{S<:LazySet} <: LazySet\n\nType that represents the action of an exponential map on a convex set.\n\nFields\n\nspmexp – sparse matrix exponential\nX      – convex set\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.dim-Tuple{LazySets.ExponentialMap}",
    "page": "Common Set Operations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(em::ExponentialMap)::Int\n\nReturn the dimension of an exponential map.\n\nInput\n\nem – an ExponentialMap\n\nOutput\n\nThe ambient dimension of the exponential map.\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.ExponentialMap}",
    "page": "Common Set Operations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d::AbstractVector{Float64}, em::ExponentialMap)::AbstractVector{Float64}\n\nReturn the support vector of the exponential map.\n\nInput\n\nd  – direction\nem – exponential map\n\nOutput\n\nThe support vector in the given direction. If the direction has norm zero, the result depends on the wrapped set.\n\nNotes\n\nIf E = exp(M)S, where M is a matrix and S is a convex set, it follows that (d E) = exp(M)(exp(M)^T d S) for any direction d.\n\n\n\n"
},

{
    "location": "lib/operations.html#Base.:∈-Tuple{AbstractArray{Float64,1},LazySets.ExponentialMap{#s1} where #s1<:LazySets.LazySet}",
    "page": "Common Set Operations",
    "title": "Base.:∈",
    "category": "Method",
    "text": "∈(x::AbstractVector{<:Real}, em::ExponentialMap{<:LazySet})::Bool\n\nCheck whether a given point is contained in an exponential map of a convex set.\n\nInput\n\nx  – point/vector\nem – linear map of a convex set\n\nOutput\n\ntrue iff x  em.\n\nAlgorithm\n\nThis implementation exploits that x  exp(M)S iff exp(-M)x  S. This follows from exp(-M)exp(M) = I for any M.\n\nExamples\n\njulia> em = ExponentialMap(SparseMatrixExp(SparseMatrixCSC([2.0 0.0; 0.0 1.0])),\n                           BallInf([1., 1.], 1.));\n\njulia> ∈([5.0, 1.0], em)\nfalse\njulia> ∈([1.0, 1.0], em)\ntrue\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.ExponentialProjectionMap",
    "page": "Common Set Operations",
    "title": "LazySets.ExponentialProjectionMap",
    "category": "Type",
    "text": "ExponentialProjectionMap{S<:LazySet} <: LazySet\n\nType that represents the application of a projection of a sparse matrix exponential to a convex set.\n\nFields\n\nspmexp – projection of a sparse matrix exponential\nX      – convex set\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.dim-Tuple{LazySets.ExponentialProjectionMap}",
    "page": "Common Set Operations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(eprojmap::ExponentialProjectionMap)::Int\n\nReturn the dimension of a projection of an exponential map.\n\nInput\n\neprojmap – projection of an exponential map\n\nOutput\n\nThe ambient dimension of the projection of an exponential map.\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.ExponentialProjectionMap}",
    "page": "Common Set Operations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d::AbstractVector{Float64}, eprojmap::ExponentialProjectionMap)::AbstractVector{Float64}\n\nReturn the support vector of a projection of an exponential map.\n\nInput\n\nd        – direction\neprojmap – projection of an exponential map\n\nOutput\n\nThe support vector in the given direction. If the direction has norm zero, the result depends on the wrapped set.\n\nNotes\n\nIf S = (LMR)X, where L and R are matrices, M is a matrix exponential, and X is a set, it follows that (d S) = LMR(R^TM^TL^Td X) for any direction d.\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.SparseMatrixExp",
    "page": "Common Set Operations",
    "title": "LazySets.SparseMatrixExp",
    "category": "Type",
    "text": "SparseMatrixExp{N<:Real}\n\nType that represents the matrix exponential, exp(M), of a sparse matrix.\n\nFields\n\nM – sparse matrix\n\nNotes\n\nThis type is provided for use with very large and very sparse matrices. The evaluation of the exponential matrix action over vectors relies on the Expokit package.\n\n\n\n"
},

{
    "location": "lib/operations.html#Base.:*-Tuple{LazySets.SparseMatrixExp,LazySets.LazySet}",
    "page": "Common Set Operations",
    "title": "Base.:*",
    "category": "Method",
    "text": "    *(spmexp::SparseMatrixExp, X::LazySet)::ExponentialMap\n\nReturn the exponential map of a convex set from a sparse matrix exponential.\n\nInput\n\nspmexp – sparse matrix exponential\nX      – convex set\n\nOutput\n\nThe exponential map of the convex set.\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.ProjectionSparseMatrixExp",
    "page": "Common Set Operations",
    "title": "LazySets.ProjectionSparseMatrixExp",
    "category": "Type",
    "text": "ProjectionSparseMatrixExp{N<:Real}\n\nType that represents the projection of a sparse matrix exponential, i.e., Lexp(M)R for a given sparse matrix M.\n\nFields\n\nL – left multiplication matrix\nE – sparse matrix exponential\nR – right multiplication matrix\n\n\n\n"
},

{
    "location": "lib/operations.html#Base.:*-Tuple{LazySets.ProjectionSparseMatrixExp,LazySets.LazySet}",
    "page": "Common Set Operations",
    "title": "Base.:*",
    "category": "Method",
    "text": "    *(projspmexp::ProjectionSparseMatrixExp, X::LazySet)::ExponentialProjectionMap\n\nReturn the application of a projection of a sparse matrix exponential to a convex set.\n\nInput\n\nprojspmexp – projection of a sparse matrix exponential\nX          – convex set\n\nOutput\n\nThe application of the projection of a sparse matrix exponential to the convex set.\n\n\n\n"
},

{
    "location": "lib/operations.html#Exponential-Map-1",
    "page": "Common Set Operations",
    "title": "Exponential Map",
    "category": "section",
    "text": "ExponentialMap\ndim(::ExponentialMap)\nσ(::AbstractVector{Float64}, ::ExponentialMap)\n∈(::AbstractVector{Float64}, ::ExponentialMap{<:LazySet})ExponentialProjectionMap\ndim(::ExponentialProjectionMap)\nσ(::AbstractVector{Float64}, ::ExponentialProjectionMap)SparseMatrixExp\n*(::SparseMatrixExp, ::LazySet)ProjectionSparseMatrixExp\n*(::ProjectionSparseMatrixExp, ::LazySet)"
},

{
    "location": "lib/operations.html#LazySets.ConvexHull",
    "page": "Common Set Operations",
    "title": "LazySets.ConvexHull",
    "category": "Type",
    "text": "ConvexHull{S1<:LazySet, S2<:LazySet} <: LazySet\n\nType that represents the convex hull of the union of two convex sets.\n\nFields\n\nX – convex set\nY – convex set\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.CH",
    "page": "Common Set Operations",
    "title": "LazySets.CH",
    "category": "Type",
    "text": "CH\n\nAlias for ConvexHull.\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.dim-Tuple{LazySets.ConvexHull}",
    "page": "Common Set Operations",
    "title": "LazySets.dim",
    "category": "Method",
    "text": "dim(ch::ConvexHull)::Int\n\nReturn the dimension of a convex hull of two convex sets.\n\nInput\n\nch – convex hull of two convex sets\n\nOutput\n\nThe ambient dimension of the convex hull of two convex sets.\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.σ-Tuple{AbstractArray{Float64,1},LazySets.ConvexHull}",
    "page": "Common Set Operations",
    "title": "LazySets.σ",
    "category": "Method",
    "text": "σ(d::AbstractVector{<:Real}, ch::ConvexHull)::AbstractVector{<:Real}\n\nReturn the support vector of a convex hull of two convex sets in a given direction.\n\nInput\n\nd  – direction\nch – convex hull of two convex sets\n\n\n\n"
},

{
    "location": "lib/operations.html#Convex-Hull-1",
    "page": "Common Set Operations",
    "title": "Convex Hull",
    "category": "section",
    "text": "ConvexHull\nCH\ndim(::ConvexHull)\nσ(::AbstractVector{Float64}, ::ConvexHull)"
},

{
    "location": "lib/operations.html#LazySets.convex_hull",
    "page": "Common Set Operations",
    "title": "LazySets.convex_hull",
    "category": "Function",
    "text": "convex_hull(points::Vector{S}; [algorithm]::String=\"monotone_chain\")::Vector{S} where {S<:AbstractVector{N}} where {N<:Real}\n\nCompute the convex hull of points in the plane.\n\nInput\n\npoints    – list of 2D vectors\nalgorithm – (optional, default: \"monotone_chain\") the convex hull                algorithm, valid options are:\n\"monotone_chain\"\n\nOutput\n\nThe convex hull as a list of 2D vectors with the coordinates of the points.\n\nExamples\n\nCompute the convex hull of a random set of points:\n\njulia> points = [randn(2) for i in 1:30]; # 30 random points in 2D\n\njulia> hull = convex_hull(points);\n\njulia> typeof(hull)\nArray{Array{Float64,1},1}\n\nPlot both the random points and the computed convex hull polygon:\n\njulia> using Plots;\n\njulia> plot([Tuple(pi) for pi in points], seriestype=:scatter);\n\njulia> plot!(VPolygon(hull), alpha=0.2);\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.convex_hull!",
    "page": "Common Set Operations",
    "title": "LazySets.convex_hull!",
    "category": "Function",
    "text": "convex_hull!(points::Vector{S}; [algorithm]::String=\"monotone_chain\")::Vector{S} where {S<:AbstractVector{N}} where {N<:Real}\n\nCompute the convex hull of points in the plane, in-place.\n\nInput\n\npoints    – list of 2D vectors (is modified)\nalgorithm – (optional, default: \"monotone_chain\") the convex hull                algorithm, valid options are:\n\"monotone_chain\"\n\nOutput\n\nThe convex hull as a list of 2D vectors with the coordinates of the points.\n\nNotes\n\nSee the non-modifying version convex_hull for more details.\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.right_turn",
    "page": "Common Set Operations",
    "title": "LazySets.right_turn",
    "category": "Function",
    "text": "right_turn(O::AbstractVector{N}, A::AbstractVector{N}, B::AbstractVector{N})::N where {N<:Real}\n\nDetermine if the acute angle defined by the three points O, A, B in the plane is a right turn (counter-clockwise) with respect to the center O.\n\nInput\n\nO – 2D center point\nA – 2D one point\nB – 2D another point\n\nOutput\n\nScalar representing the rotation.\n\nAlgorithm\n\nThe cross product is used to determine the sense of rotation. If the result is 0, the points are collinear; if it is positive, the three points constitute a positive angle of rotation around O from A to B; otherwise they constitute a negative angle.\n\n\n\n"
},

{
    "location": "lib/operations.html#LazySets.monotone_chain!",
    "page": "Common Set Operations",
    "title": "LazySets.monotone_chain!",
    "category": "Function",
    "text": "monotone_chain!(points::Vector{S})::Vector{S} where {S<:AbstractVector{N}} where {N<:Real}\n\nCompute the convex hull of points in the plane using Andrew's monotone chain method.\n\nInput\n\npoints – list of 2D vectors; is sorted in-place inside this function\n\nOutput\n\nList of vectors containing the 2D coordinates of the corner points of the convex hull.\n\nNotes\n\nFor large sets of points, it is convenient to use static vectors to get maximum performance. For information on how to convert usual vectors into static vectors, see the type SVector provided by the StaticArrays package.\n\nAlgorithm\n\nThis function implements Andrew's monotone chain convex hull algorithm to construct the convex hull of a set of n points in the plane in O(n log n) time. For further details see Monotone chain\n\n\n\n"
},

{
    "location": "lib/operations.html#Convex-Hull-Algorithms-1",
    "page": "Common Set Operations",
    "title": "Convex Hull Algorithms",
    "category": "section",
    "text": "convex_hull\nconvex_hull!\nright_turn\nmonotone_chain!"
},

{
    "location": "lib/approximations.html#",
    "page": "Approximations",
    "title": "Approximations",
    "category": "page",
    "text": ""
},

{
    "location": "lib/approximations.html#LazySets.Approximations",
    "page": "Approximations",
    "title": "LazySets.Approximations",
    "category": "Module",
    "text": "Module Approximations.jl – polygonal approximation of convex sets through support vectors.\n\n\n\n"
},

{
    "location": "lib/approximations.html#Approximations-1",
    "page": "Approximations",
    "title": "Approximations",
    "category": "section",
    "text": "This section of the manual describes the Cartesian decomposition algorithms and the approximation of high-dimensional convex sets using projections.Pages = [\"approximations.md\"]\nDepth = 3CurrentModule = LazySets.Approximations\nDocTestSetup = quote\n    using LazySets, LazySets.Approximations\nendApproximations"
},

{
    "location": "lib/approximations.html#LazySets.Approximations.decompose",
    "page": "Approximations",
    "title": "LazySets.Approximations.decompose",
    "category": "Function",
    "text": "decompose(S::LazySet)::CartesianProductArray\n\nCompute an overapproximation of the projections of the given convex set over each two-dimensional subspace using box directions.\n\nInput\n\nS – convex set\n\nOutput\n\nA CartesianProductArray corresponding to the Cartesian product of 2 times 2 box-shaped polygons.\n\n\n\ndecompose(S::LazySet, ɛi::Vector{Float64})::CartesianProductArray\n\nCompute an overapproximation of the projections of the given convex set over each two-dimensional subspace with a certified error bound.\n\nInput\n\nS  – convex set\nɛi – array with the error bound for each projection (different error bounds         can be passed for different blocks)\n\nOutput\n\nA CartesianProductArray corresponding to the Cartesian product of 2 times 2 polygons.\n\nAlgorithm\n\nThis algorithm assumes a decomposition into two-dimensional subspaces only, i.e., partitions of the form 2 2  2. In particular, if S is a CartesianProductArray, no check is performed to verify that assumption.\n\nThe algorithm proceeds as follows:\n\nProject the set S into each partition, with M⋅S, where M is the identity matrix in the block coordinates and zero otherwise.\nOverapproximate the set with a given error bound, ɛi[i], for i = 1  b,\nReturn the result as a CartesianProductArray.\n\n\n\ndecompose(S::LazySet, ɛ::Float64)::CartesianProductArray\n\nCompute an overapproximation of the projections of the given convex set over each two-dimensional subspace with a certified error bound.\n\nInput\n\nS – convex set\nɛ –  error bound\n\nOutput\n\nA CartesianProductArray corresponding to the Cartesian product of 2 times 2 polygons.\n\nNotes\n\nThis function is a particular case of decompose(S, ɛi), where the same error bound for each block is assumed.\n\n\n\n"
},

{
    "location": "lib/approximations.html#LazySets.Approximations.overapproximate",
    "page": "Approximations",
    "title": "LazySets.Approximations.overapproximate",
    "category": "Function",
    "text": "overapproximate(S::LazySet)::HPolygon\n\nReturn an approximation of a given 2D convex set as a box-shaped polygon.\n\nInput\n\nS – convex set, assumed to be two-dimensional\n\nOutput\n\nA box-shaped polygon in constraint representation.\n\n\n\noverapproximate(S::LazySet, ɛ::Float64)::HPolygon\n\nReturn an ɛ-close approximation of the given 2D set (in terms of Hausdorff distance) as a polygon.\n\nInput\n\nS – convex set, assumed to be two-dimensional\nɛ – error bound\n\nOutput\n\nA polygon in constraint representation.\n\n\n\n"
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
    "text": "ballinf_approximation(S)\n\nOverapproximate a convex set by a tight ball in the infinity norm.\n\nInput\n\nS – convex set\n\nOutput\n\nA tight ball in the infinity norm.\n\nAlgorithm\n\nThe center and radius of the box are obtained by evaluating the support function of the given convex set along the canonical directions.\n\n\n\n"
},

{
    "location": "lib/approximations.html#LazySets.Approximations.box_approximation",
    "page": "Approximations",
    "title": "LazySets.Approximations.box_approximation",
    "category": "Function",
    "text": "box_approximation(S::LazySet)::Hyperrectangle\n\nOverapproximate a convex set by a tight hyperrectangle.\n\nInput\n\nS – convex set\n\nOutput\n\nA tight hyperrectangle.\n\nAlgorithm\n\nThe center of the hyperrectangle is obtained by averaging the support function of the given set in the canonical directions, and the lengths of the sides can be recovered from the distance among support functions in the same directions.\n\n\n\n"
},

{
    "location": "lib/approximations.html#LazySets.Approximations.interval_hull",
    "page": "Approximations",
    "title": "LazySets.Approximations.interval_hull",
    "category": "Function",
    "text": "interval_hull\n\nAlias for box_approximation.\n\n\n\n"
},

{
    "location": "lib/approximations.html#LazySets.Approximations.box_approximation_symmetric",
    "page": "Approximations",
    "title": "LazySets.Approximations.box_approximation_symmetric",
    "category": "Function",
    "text": "box_approximation_symmetric(S::LazySet)::Hyperrectangle\n\nOverapproximate a convex set by a tight hyperrectangle centered in the origin.\n\nInput\n\nS – convex set\n\nOutput\n\nA tight hyperrectangle centered in the origin.\n\nAlgorithm\n\nThe center of the box is the origin, and the radius is obtained by computing the maximum value of the support function evaluated at the canonical directions.\n\n\n\n"
},

{
    "location": "lib/approximations.html#LazySets.Approximations.symmetric_interval_hull",
    "page": "Approximations",
    "title": "LazySets.Approximations.symmetric_interval_hull",
    "category": "Function",
    "text": "symmetric_interval_hull\n\nAlias for box_approximation_symmetric.\n\n\n\n"
},

{
    "location": "lib/approximations.html#LazySets.Approximations.box_approximation_helper",
    "page": "Approximations",
    "title": "LazySets.Approximations.box_approximation_helper",
    "category": "Function",
    "text": "box_approximation_helper(S::LazySet)\n\nCommon code of box_approximation and box_approximation_symmetric.\n\nInput\n\nS – convex set\n\nOutput\n\nA tuple containing the data that is needed to construct a tightly overapproximating hyperrectangle.\n\nc – center\nr – radius\n\nAlgorithm\n\nThe center of the hyperrectangle is obtained by averaging the support function of the given convex set in the canonical directions. The lengths of the sides can be recovered from the distance among support functions in the same directions.\n\n\n\n"
},

{
    "location": "lib/approximations.html#Box-Approximations-1",
    "page": "Approximations",
    "title": "Box Approximations",
    "category": "section",
    "text": "ballinf_approximation\nbox_approximation\ninterval_hull\nbox_approximation_symmetric\nsymmetric_interval_hull\nbox_approximation_helper"
},

{
    "location": "lib/approximations.html#Base.LinAlg.norm",
    "page": "Approximations",
    "title": "Base.LinAlg.norm",
    "category": "Function",
    "text": "norm(S::LazySet, [p]::Real=Inf)\n\nReturn the norm of a convex set. It is the norm of the enclosing ball (of the given norm) of minimal volume.\n\nInput\n\nS – convex set\np – (optional, default: Inf) norm\n\nOutput\n\nA real number representing the norm.\n\n\n\n"
},

{
    "location": "lib/approximations.html#LazySets.radius",
    "page": "Approximations",
    "title": "LazySets.radius",
    "category": "Function",
    "text": "radius(S::LazySet, [p]::Real=Inf)\n\nReturn the radius of a convex set. It is the radius of the enclosing ball (of the given norm) of minimal volume with the same center.\n\nInput\n\nS – convex set\np – (optional, default: Inf) norm\n\nOutput\n\nA real number representing the radius.\n\n\n\n"
},

{
    "location": "lib/approximations.html#LazySets.diameter",
    "page": "Approximations",
    "title": "LazySets.diameter",
    "category": "Function",
    "text": "diameter(S::LazySet, [p]::Real=Inf)\n\nReturn the diameter of a convex set. It is the maximum distance between any two elements of the set, or, equivalently, the diameter of the enclosing ball (of the given norm) of minimal volume with the same center.\n\nInput\n\nS – convex set\np – (optional, default: Inf) norm\n\nOutput\n\nA real number representing the diameter.\n\n\n\n"
},

{
    "location": "lib/approximations.html#Metric-properties-of-sets-1",
    "page": "Approximations",
    "title": "Metric properties of sets",
    "category": "section",
    "text": "norm(::LazySet, ::Real=Inf)\nradius(::LazySet, ::Real=Inf)\ndiameter(::LazySet, ::Real=Inf)"
},

{
    "location": "lib/approximations.html#LazySets.Approximations.approximate",
    "page": "Approximations",
    "title": "LazySets.Approximations.approximate",
    "category": "Function",
    "text": "approximate(S::LazySet, ɛ::Float64)::Vector{Approximation2D}\n\nReturn an ɛ-close approximation of the given 2D convex set (in terms of Hausdorff distance) as an inner and an outer approximation composed by sorted local Approximation2D.\n\nInput\n\nS – 2D convex set\nɛ – error bound\n\nOutput\n\nAn ɛ-close approximation of the given 2D convex set.\n\n\n\n"
},

{
    "location": "lib/approximations.html#Iterative-refinement-1",
    "page": "Approximations",
    "title": "Iterative refinement",
    "category": "section",
    "text": "approximateSee Iterative Refinement for more details."
},

{
    "location": "lib/utils.html#",
    "page": "Utility Functions",
    "title": "Utility Functions",
    "category": "page",
    "text": "CurrentModule = LazySets\nDocTestSetup = quote\n    using LazySets\nend"
},

{
    "location": "lib/utils.html#LazySets.sign_cadlag",
    "page": "Utility Functions",
    "title": "LazySets.sign_cadlag",
    "category": "Function",
    "text": "sign_cadlag(x::N)::N where {N<:Real}\n\nThis function works like the sign function but is 1 for input 0.\n\nInput\n\nx – real scalar\n\nOutput\n\n1 if x  0, -1 otherwise.\n\nNotes\n\nThis is the sign function right-continuous at zero (see càdlàg function). It can be used with vector-valued arguments via the dot operator.\n\nExamples\n\njulia> sign_cadlag.([-0.6, 1.3, 0.0])\n3-element Array{Float64,1}:\n -1.0\n  1.0\n  1.0\n\n\n\n"
},

{
    "location": "lib/utils.html#LazySets.jump2pi",
    "page": "Utility Functions",
    "title": "LazySets.jump2pi",
    "category": "Function",
    "text": "jump2pi(x::Float64)::Float64\n\nReturn x + 2 if x is negative, otherwise return x.\n\nInput\n\nx – real scalar\n\nOutput\n\nx + 2 if x is negative, x otherwise.\n\nExamples\n\njulia> jump2pi(0.0)\n0.0\njulia> jump2pi(-0.5)\n5.783185307179586\njulia> jump2pi(0.5)\n0.5\n\n\n\n"
},

{
    "location": "lib/utils.html#Base.:<=-Tuple{AbstractArray{Float64,1},AbstractArray{Float64,1}}",
    "page": "Utility Functions",
    "title": "Base.:<=",
    "category": "Method",
    "text": "<=(u::AbstractVector{Float64}, v::AbstractVector{Float64})::Bool\n\nCompares two 2D vectors by their direction.\n\nInput\n\nu –  first 2D direction\nv –  second 2D direction\n\nOutput\n\nTrue iff arg(u) 2  arg(v) 2\n\nNotes\n\nThe argument is measured in counter-clockwise fashion, with the 0 being the direction (1, 0).\n\n\n\n"
},

{
    "location": "lib/utils.html#Utility-functions-1",
    "page": "Utility Functions",
    "title": "Utility functions",
    "category": "section",
    "text": "sign_cadlag\njump2pi\n<=(::AbstractVector{Float64}, ::AbstractVector{Float64})"
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
    "text": "If you like this package, consider contributing! You can send bug reports (or fix them and send your code), add examples to the documentation, or propose new features.Below some conventions that we follow when contributing to this package are detailed. For specific guidelines on documentation, see the Documentations Guidelines wiki."
},

{
    "location": "about.html#Branches-and-pull-requests-(PR)-1",
    "page": "About",
    "title": "Branches and pull requests (PR)",
    "category": "section",
    "text": "We use a standard pull request policy: You work in a private branch and eventually add a pull request, which is then reviewed by other programmers and merged into the master branch.Each pull request should be pushed in a new branch with the name of the author followed by a descriptive name, e.g., mforets/my_feature. If the branch is associated to a previous discussion in one issue, we use the name of the issue for easier lookup, e.g., mforets/7."
},

{
    "location": "about.html#Unit-testing-and-continuous-integration-(CI)-1",
    "page": "About",
    "title": "Unit testing and continuous integration (CI)",
    "category": "section",
    "text": "This project is synchronized with Travis CI such that each PR gets tested before merging (and the build is automatically triggered after each new commit). For the maintainability of this project, it is important to understand and fix the failing doctests if they exist. We develop in Julia v0.6.0, but for experimentation we also build on the nightly branch.When you modify code in this package, you should make sure that all unit tests pass. To run the unit tests locally, you should do:$ julia --color=yes test/runtests.jlAlternatively, you can achieve the same from inside the REPL using the following command:julia> Pkg.test(\"LazySets\")We also advise adding new unit tests when adding new features to ensure long-term support of your contributions."
},

{
    "location": "about.html#Contributing-to-the-documentation-1",
    "page": "About",
    "title": "Contributing to the documentation",
    "category": "section",
    "text": "Documenter.jl :New functions and types should be documented according to our guidelines directly in the source code.You can view the source code documentation from inside the REPL by typing ? followed by the name of the type or function. For example, the following command will print the documentation of the LazySet type:julia> ?LazySetThis documentation you are currently reading is written in Markdown, and it relies on Documenter.jl to produce the HTML layout. The sources for creating this documentation are found in docs/src. You can easily include the documentation that you wrote for your functions or types there (see the Documenter.jl guide or our sources for examples).To generate the documentation locally, run make.jl, e.g., by executing the following command in the terminal:$ julia --color=yes docs/make.jlNote that this also runs all doctests which will take some time."
},

{
    "location": "about.html#Related-projects-1",
    "page": "About",
    "title": "Related projects",
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
