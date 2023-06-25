
% ========================================================================
% INFO: 
% This script was used to calulate the Dice Coefficient. It is based on the
% Matlab-function "dice()", but adds the value +eps() to numerator and
% denominator to avoid division by zero in case of fully closed vocal
% folds.
% Further information can be found in our paper: 
% Fehling, M.K., Grosch, F., Schuster, M.E., Schick, B., Lohscheller, J., 2020.
% Fully automatic segmentation of glottis and vocal folds in endoscopic laryngealhigh-speed videos using a deep Convolutional LSTM Network." 
% PloS one. DOI: 10.1371/journal.pone.0227791
% [MKF]
% ========================================================================

function similarity = own_dice(A,B)

    try

        inter = nnz(A & B);
        union = nnz(A | B);
        jac = (inter + eps) / (union + eps);

    catch ME
        throw(ME)
    end

    similarity = 2 * jac ./ (1 + jac + eps);
    

end


