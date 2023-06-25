
% ========================================================================
% INFO: 
% This function calculates the points P1 to P4. 
%
% [MKF]
% ========================================================================

function [P1, P2, P3, P4] = calc_points_Pi(Seg, M, gA, gAs) 

    % --- some definitions: 
    
        factLen = 1.125; 
        factWid = 0.25;

        % PN and AN 
        PN = M - round( factLen * 0.5 * gA );
        AN = M + round( factLen * 0.5 * gA );

        % ensure that PN and AN are \in [1, 256]: 
        AN(AN<1) = 1;      AN(AN>256) = 256;
        PN(PN<1) = 1;      PN(PN>256) = 256;

        % MA and MP
        MA = M + round(1/6 * gA);
        MP = M - round(1/6 * gA);

        MMr = M - gAs; 
        MMl = M + gAs;


    % --- points P1, P2, P3 and P4 analog to 2007_Lohscheller_MedIA
    
        % IDEA: 
        % Find intersection between segmentation and corresponding mask, 
        % use min and max to determine P1 & P2 resp. P3 & P4 
        
        % P1: dorsal ending of the glottal main axis,
        % P2: ventral ending of the glottal main axis,
        % P3: right vocal fold edge at medial position,
        % P4: left vocal fold edge at medial position.

        % matrices for axes
        M_gA = lineCoords(zeros(256,256), PN, AN);    
        M_gAs  = lineCoords(zeros(256,256), MMl, MMr);   

        
        % add axes
        tmp = Seg + M_gA;
        Int_gA = zeros(256, 256);
        Int_gA(tmp==2) = 1;

        tmp = Seg + M_gAs;
        Int_gAs = zeros(256, 256);
        Int_gAs(tmp==2) = 1;

            clear tmp

            
        % find min and max
        
        [val_rowmin, val_colmin, val_rowmax, val_colmax] = findP1P2onSeg(Seg);
            P1 = [val_rowmin; val_colmin];
            P2 = [val_rowmax; val_colmax];

            clear val_rowmin val_colmin val_rowmax val_colmax


        % exception for P3 & P4: These points are (fix) defined at the 
        % middle of the vocal folds, so they might not exist in all
        % all segmentations, i.e. in case of small glottis. If they do
        % not exist, set the coordinates to [0,0].
        
        if sum(sum(Int_gAs)) == 0
            P3 = [0;0];
            P4 = [0;0];

        else
            [tmp_r, tmp_c] = find(Int_gAs == 1);

            P3 = [tmp_r(1,1); tmp_c(1,1)];      % VF r
            P4 = [tmp_r(size(tmp_r,1),1); tmp_c(size(tmp_r,1),1)]; % VF l

            clear tmp_r tmp_c
        end                 
