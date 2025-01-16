function [sensor,num_Lst,InputFile,main_meta] = LoadSensorType(path_data)
%LOADSENSORTYPE Basic metadata should be loaded first to see which sensor
%here.
% Add Landsat 9 , Feb., 17, 2022

%% Search metadate file for Landsat 4-8
    main_meta=dir(fullfile(path_data,'L*MTL.txt'));
    existMTL=size(main_meta);
    InputFile=[];
    if existMTL(1)==0
        main_meta=dir(fullfile(path_data, 'S*MTD*TL*.xml'));
        if isempty(main_meta)
            main_meta=dir(fullfile(path_data, 'MTD*TL*.xml'));
        end
        if ~isempty(main_meta)
            txtstart = strfind(path_data,'S2A') ; 
            num_Lst='2A';
            if isempty(txtstart)
                txtstart = strfind(path_data,'S2B') ;    % S2A or S2B  by Shi 10/18/2017
                num_Lst='2B';
            end
            if isempty(txtstart)
                txtstart = strfind(path_data,'S2C') ;    % or S2C by Shi 1/16/2025
                num_Lst='2C';
            end
            txtend   = strfind(path_data,'.SAFE')-1 ;   
            InputFile.DataStrip = path_data(txtstart:txtend) ;

            InputFile.Dmain = path_data(1:txtstart-1) ;

            txtstart = strfind(path_data,'GRANULE')+8 ;
            
            InputFile.InspireXML=path_data(1:txtstart-10); % extra add to read inspire xml.
            
            txtend   =  length(path_data);
            InputFile.Granule = path_data(txtstart:txtend) ;
            InputFile.pathh = path_data ;
            main_meta.name=InputFile.Granule;% return file name fro Fmask results.
        else
%             fprintf('No available data in the current folder!\n');
            sensor=[]; % no supported image.
            num_Lst=[];
            InputFile=[];
            main_meta=[];
            return;
        end
    else
        % determine sensor type
        % open and read hdr file
        fid_in=fopen(fullfile(path_data,main_meta.name),'r');
        geo_char=fscanf(fid_in,'%c',inf);
        fclose(fid_in);
        geo_char=geo_char';
        geo_str=strread(geo_char,'%s');

        % Identify Landsat Number (Lnum = 4, 5, 7, or 8)
        LID=char(geo_str(strmatch('SPACECRAFT_ID',geo_str)+2));
%         num_Lst=str2double(LID(end-1));
        num_Lst=(LID(end-1));
    end
    % define Landsat sensor.
    sensor='';
    if strcmp(num_Lst,'8') | strcmp(num_Lst,'9')
        sensor='L_OLI_TIRS';
    else
        if strcmp(num_Lst,'4')||strcmp(num_Lst,'5')||strcmp(num_Lst,'6')
            sensor='L_TM';
        else
            if strcmp(num_Lst,'7')
                sensor='L_ETM_PLUS';
            end
        end
    end
    if strcmp(num_Lst,'2A')||strcmp(num_Lst,'2B')||strcmp(num_Lst,'2C')
        sensor='S_MSI';
    end
end

