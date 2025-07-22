% function [Montage] = ExtractMontage (csdFileName, eegLabels)
% 
% This is a generic routine to determine any EEG montage from a *.CSD
% ASCII file (cf. Kayser & Tenke, 2006a) using an ordered list of channel
% labels (i.e., a cell string array).
%
% Usage: [ Montage ] = ExtractMontage ( csdFileName, eegLabels );
%
%   Input arguments:   csdFileName   *.CSD file name (generic CSD montage)
%                        eegLabels   cell string array with channel labels
%
%   Output argument:       Montage   cell structure consisting of a channel
%                                    label 'lab', 2-D plane x-y coordinates
%                                    'xy', and 3-D spherical angles 'theta'
%                                    and 'phi'
%        
% Copyright (C) 2007 by J?gen Kayser (Email: kayserj@pi.cpmc.columbia.edu)
% GNU General Public License (http://www.gnu.org/licenses/gpl.txt)
% Updated: $Date: 2009/05/14 17:26:00 $ $Author: jk $
%
function [montage] = MyExtractMontage(csdFileName,labelsFileName)
    %% Load csd data and label data
    fIDCSD = fopen(csdFileName,'r');
    elecData = textscan(fIDCSD,'%s %f %f %f %f %f %f %f','commentstyle','//');
    fclose(fIDCSD);
    fIDLab = fopen(labelsFileName,'r');
    eleclab = textscan(fIDLab,'%s');
    fclose(fIDLab);
    
    %% Extract electrodes
    lab = cell(length(eleclab{1}),1);
    montage = struct('lab',[],'theta',zeros(length(eleclab{1}),1),...
        'phi',zeros(length(eleclab{1}),1),'coorX',zeros(length(eleclab{1}),1),'coorY',zeros(length(eleclab{1}),1));
    for eleci = 1:length(eleclab{1})
        switch eleclab{1}{eleci}
            case 'O9'       % O9 is I1 in 10-5 system, therefore, electrode name convert to O9 from I1
                num = strcmpi('I1',elecData{1});
                lab(eleci) =  {'O9'};
            case 'O10'      % O10 is I2 in 10-5 system, therefore, electrode name convert to O10 from I2
                num = strcmpi('I2',elecData{1});
                lab(eleci) = {'O10'};
            otherwise
                num = strcmpi(eleclab{1}(eleci),elecData{1});
                lab(eleci) = elecData{1}(num);
        end
        montage.theta(eleci) = elecData{2}(num);
        montage.phi(eleci) = elecData{3}(num);
        montage.coorX(eleci) = elecData{5}(num);
        montage.coorY(eleci) = elecData{6}(num);
    end
    montage.lab = lab;
    
    %% Convert coordinate systm
    phiT = 90 - montage.phi;                    % calculate phi from top of sphere
    theta2 = (2 * pi * montage.theta) / 360;    % convert degrees to radians
    phi2 = (2 * pi * phiT) / 360;
    [x,y] = pol2cart(theta2,phi2);      % get plane coordinates
    xy = [x y];
    xy = xy/max(max(xy));               % set maximum to unit length
    xy = xy/2 + 0.5;                    % adjust to range 0-1
    montage.xy = xy;
end
