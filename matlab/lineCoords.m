
function [mat] = dev_lineCoords(mat, P1, P2)
 
% initial conditon
p1 = P1';
p2 = P2';

% use file provided on file exchange
[x, y] = bresenham(p1(1),p1(2),p2(1),p2(2));

% replace entries in matrix mat
mat(sub2ind(size(mat), y, x)) = 1;
