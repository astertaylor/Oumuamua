// Gmsh project created on Fri Feb 04 15:40:44 2022
//+
SetFactory("OpenCASCADE");
Sphere(1) = {-0, -0, 0, 1, -Pi/2, Pi/2, 2*Pi};
//+
Physical Surface(1) = {1};
