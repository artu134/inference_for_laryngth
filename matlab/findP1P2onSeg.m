
function [val_rowmin, val_colmin, val_rowmax, val_colmax] = findP1P2onSeg(SegData)

% init:
val_rowmin = 0;
val_colmin = 0; 
val_rowmax = 0;
val_colmax = 0;

% coodinates of all pixels labled as "glottis" in case of a binary mask
if sum(find(SegData >= 2)) == 0
    [coord_col, coord_row] = find(SegData == 1);
else
    % delete all "1" (encode artefacts) 
    SegData(SegData == 1) = 0;
    
    % set all others to 1
    SegData(SegData >= 2) = 1;
    
    [coord_col, coord_row] = find(SegData == 1);
end

% find colmin and colmax
val_colmin = min(coord_col);
val_colmax = max(coord_col);

% find possible values for row ("y-value")
ind_rowmin = find(coord_col == val_colmin);
ind_rowmax = find(coord_col == val_colmax);

%tmp
tmp_rowmin = [];
tmp_rowmax = [];

for i = 1:size(ind_rowmin)
    tmp_rowmin = [tmp_rowmin; coord_row(ind_rowmin(i,1),1)];
end

for i = 1:size(ind_rowmax)
    tmp_rowmax = [tmp_rowmax; coord_row(ind_rowmax(i,1),1)];
end

val_rowmin = mean(tmp_rowmin);
val_rowmax = mean(tmp_rowmax); 


% exception: in case of "empty" segmentations (= closed state) 
if isempty(val_rowmin) == 1 || isnan(val_rowmin) == 1,    val_rowmin = 0;     end
if isempty(val_colmin) == 1 || isnan(val_colmin) == 1,    val_colmin = 0;     end
if isempty(val_rowmax) == 1 || isnan(val_rowmax) == 1,    val_rowmax = 0;     end
if isempty(val_colmax) == 1 || isnan(val_colmax) == 1,    val_colmax = 0;     end

