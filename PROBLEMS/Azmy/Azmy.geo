Point(1) = {0, 0, 0, 1.0};
Point(2) = {5, 0, 0, 1.0};
Point(3) = {10, 0, 0, 1.0};
Point(4) = {0, 5, 0, 1.0};
Point(5) = {0, 10, 0, 1.0};
Point(6) = {10, 10, 0, 1.0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 6};
Line(4) = {6, 5};
Line(5) = {5, 4};
Line(6) = {4, 1};
Point(7) = {5, 5, 0, 1.0};
Line(7) = {2, 7};
Line(8) = {7, 4};
Line Loop(9) = {1, 7, 8, 6};
Plane Surface(10) = {9};
Line Loop(11) = {2, 3, 4, 5, -8, -7};
Plane Surface(12) = {11};
Physical Surface("M1") = {10};
Physical Surface("M2") = {12};
Physical Line("reflective") = {1, 2, 6, 5};
Physical Line("vacuum") = {3, 4};
