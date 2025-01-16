function Fmask_4_7(input1, input2, input3, input4, input5)
%% Standalone of Fmask 4.7 version
% Input formats
%
% 1) Set default program
%     Fmask_4_7()
%
% 2) Set buffers of cloud, shadow, and snow/ice
%    Fmask_4_7(3, 3, 1)
%
% 3) Set buffers of cloud, shadow, and snow/ice, and threshold of cloud probability
%    Fmask_4_7(3, 3, 3, 22.5)
%
%
% Below cases are to setup the directory of the auxiliary data (the folder
% <AuxiData>) for the implement which fails to locate the directory by the
% default settings. To examine this, please process an image using the
% default program (like case 1), and if a warning message 'Fail to locate
% the directory of auxiliary data'  presents, this means the default
% settings do not work, and then you need to custumize the directory of
% auxiliary data using below inputs.
%

% This problem usually occurs in Linux system.
% (a warning message 'Fail to locate the directory of auxiliary data' will presents).
%
% Note: Start up from the Matlab code will not be affected, and please
% ignore the below cases
%
% Please consider them only when a warning message 'Fail to locate the auxiliary data' present at default model
%
% 4) Set the directory of the auxiliary data 
%     Fmask_4_7('C:\Users\xxx\xxx\Fmask_4_4\AuxiData')
%
% 5) Set buffers of cloud, shadow, and snow/ice, and the auxiliary data
%    Fmask_4_7(3, 3, 0, 'C:\Users\xxx\xxx\Fmask_4_4\AuxiData')
%
% 6) Set buffers of cloud, shadow, and snow/ice, threshold of cloud probability, and the auxiliary data
%    Fmask_4_7(3, 3, 0, 22.5, 'C:\Users\xxx\xxx\Fmask_4_4\AuxiData')
%

    %% Case 1)
    if ~exist('input1', 'var')
         autoFmask(); % default buffering pixels for cloud, cloud shadow, and snow
         return;
    end
    
    %% Case 2)
    if exist('input3', 'var')&& ~exist('input4', 'var')
         autoFmask('cloud', force2num(input1),'shadow', force2num(input2), 'snow',  force2num(input3));
         return;
    end
    
    if exist('input4', 'var')&& ~exist('input5', 'var')
        if isnumeric(input4) || ~isfolder(input4)
            %% Case 3)
            autoFmask('cloud', force2num(input1),'shadow', force2num(input2), 'snow', force2num(input3), 'p', force2num(input4));
        else
            %% Case 5)
            if isfolder(fullfile(input4, 'GTOPO30ZIP')) && isfolder(fullfile(input4, 'GSWO150ZIP')) 
                autoFmask('cloud', force2num(input1),'shadow', force2num(input2), 'snow', force2num(input3), 'auxi', input4);
            else
                fprintf('Do not find the directory of the auxiliary data. Please input a correct one. \r');
            end
         end
         return;
    end

    %% Case 4)
    if exist('input1', 'var') && ~exist('input2', 'var')
        if isfolder(input1) && isfolder(fullfile(input1, 'GTOPO30ZIP')) && isfolder(fullfile(input1, 'GSWO150ZIP')) 
            autoFmask('auxi', input1);
        else
            fprintf('Do not find the directory of the auxiliary data. Please correct it. \r');
        end
         return;
    end

    %% Case 6)
    if exist('input5', 'var')
    %     cldpix =input1; sdpix = input2; snpix = input3; cldprob = input4; pathauxi = input5; 
        if isfolder(input5) && isfolder(fullfile(input5, 'GTOPO30ZIP')) && isfolder(fullfile(input5, 'GSWO150ZIP')) 
            autoFmask('cloud',force2num(input1),'shadow',force2num(input2),'snow',force2num(input3),'p',force2num(input4), 'auxi', input5);
        else
            fprintf('Do not find the directory of the auxiliary data. Please correct it. \r');
        end
        return;
    end
end

function input2 = force2num(input2)
%% convert to number input if not
        if ~isnumeric(input2)
            input2 = str2num(input2);
        end
end

% old version
%
%     if exist('cldpix','var')==1&&exist('sdpix','var')==1&&exist('snpix','var')==1
%         if exist('cldprob','var')==1
%             autoFmask('cloud',str2num(cldpix),'shadow',str2num(sdpix),'snow',str2num(snpix),'p',str2num(cldprob));
%         else
%             autoFmask('cloud',str2num(cldpix),'shadow',str2num(sdpix),'snow',str2num(snpix));
%         end
%     else
%         % default buffering pixels for cloud, cloud shadow, and snow
%         autoFmask();
%     end
