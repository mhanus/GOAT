Point(1) = {-10, -10, 0, 1.0};
Point(2) = {10, -10, 0, 1.0};
Point(3) = {-5, -5, 0, 0.5};
Point(4) = {5, -5, 0, 0.5};
Point(5) = {-5, 5, 0, 0.5};
Point(6) = {5, 5, 0, 0.5};
Point(7) = {-10, 10, 0, 1.0};
Point(8) = {10, 10, 0, 1.0};
Line(1) = {1, 2};
Line(2) = {2, 8};
Line(3) = {8, 7};
Line(4) = {7, 1};
Line(5) = {3, 4};
Line(6) = {4, 6};
Line(7) = {6, 5};
Line(8) = {5, 3};
Line Loop(9) = {1, 2, 3, 4};
Line Loop(10) = {7, 8, 5, 6};
Point(11) = {5, -7, 0, 0.5};
Point(12) = {7, -7, 0, 0.5};
Point(13) = {7, 7, 0, 0.5};
Point(14) = {5, 7, 0, 0.5};
Physical Line("S") = {1};
Physical Line("E") = {2};
Physical Line("N") = {3};
Physical Line("W") = {4};
Line(11) = {4, 11};
Line(12) = {11, 12};
Line(13) = {12, 13};
Line(14) = {13, 14};
Line(15) = {14, 6};
Plane Surface(16) = {10};
Line Loop(17) = {8, 5, 11, 12, 13, 14, 15, 7};
Plane Surface(18) = {9, 17};
Line Loop(19) = {15, -6, 11, 12, 13, 14};
Plane Surface(20) = {19};
Physical Surface("M1") = {16};
Physical Surface("M2") = {18};
Physical Surface("D") = {20};
