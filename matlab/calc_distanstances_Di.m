
% ========================================================================
% INFO: 
% This function calculates the Distances Di (i = 1,2,3,4) from the points 
% Pi_GT and Pi_NN derived from the Ground Truth and the Neural Network
% segmentation, respectively.
% In case that the point Pi isn't defined in one of the corresponding
% segmentations, the distance is set to -1. 
% 
% [MKF]
% ========================================================================

function [D1, D2, D3, D4] = calc_distanstances_Di(P1_GT, P1_NN, P2_GT, P2_NN, P3_GT, P3_NN, P4_GT, P4_NN)

    % --- calculate distances analog to 2007_Lohscheller_MedIA

        if sum(P1_GT) == 0 || sum(P1_NN) == 0
            % P1 is not defined for at least one segmentation result
            D1 =  -1;                    
        else
            D1 = norm(P1_GT - P1_NN);              
        end
        
        
        if sum(P2_GT) == 0 || sum(P2_NN) == 0
            % P2 is not defined for at least one segmentation result
            D2 =  -1;                    
        else
            D2 = norm(P2_GT - P2_NN);            
        end


        if sum(P3_GT) == 0 || sum(P3_NN) == 0
            % P3 is not defined for at least one segmentation result
            D3 =  -1;                    
        else
            D3 = norm(P3_GT - P3_NN);                 
        end


        if sum(P4_GT) == 0 || sum(P4_NN) == 0
            % P4 is not defined for at least one segmentation result
            D4 = -1;
        else
            D4 = norm(P4_GT - P4_NN);     
        end

end

