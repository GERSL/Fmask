function clr_pct = autoFmask(varargin)
% AUTOFMASK Automatedly detect clouds, cloud shadows, snow, and water for
%     Landsats 4-7 TM/EMT+, Landsat 8 OLI/TIRS, and Sentinel 2 MSI images.
%
%
% Description
%
%     This 4.7 version has better cloud, cloud shadow, and snow detection
%     results for Sentinel-2 data and better results (compared to the 3.3
%     version that is being used by USGS as the Colection 1 QA Band) for
%     Landsats 4-8 data as well.
%     
%     If any questions, please do not hesitate to contact 
%     Shi Qiu (shi.qiu@uconn.edu) and Zhe Zhu (zhe@uconn.edu)
%
%
% Input arguments
%
%     cloud     Dilated number of pixels for cloud with default value of 3.
%     shadow    Dilated number of pixels for cloud shadow with default value of 3.
%     snow      Dilated number of pixels for snow with default value of 0.
%     p         Cloud probability threshold with default values of 10.0 for
%               Landsats 4~7, 17.5 for Landsat 8, and 20.0 for Sentinel 2.
%     d         Radius of dilation for Potential False Positive Cloud such as
%               urban/built-up and (mountian) snow/ice. Default: 0 meters.
%               Higher the value, Larger the potential false positive cloud
%               layer. This is used for the places where the orginal Potential 
%               False Positive Cloud Layer fails to include the false
%               posistive clouds.
%     e         Radius of erosion for Potential False Positive Cloud such as
%               urban/built-up and (mountian) snow/ice. Default: 150 meters
%               for Landsats 4-7 and 90 meters for Landsat 8 and
%               Sentinel-2.
%     sw      ShadowWater (SW) means the shadow of cloud over water.
%               Default: False
%               We do not suggest mask out the cloud shadow over water
%               since it is less meanful and very time-comsuing.
%     udem      The path of User's DEM data. (.tiff). If users provide
%               local DEM data, Fmask 4.1 will process the image along with this DEM
%               data; or, the default USGS GTOPO30 will be used.
%     auxi      The path of the auxiliary data ('AuxiData'). (for standalone only)
%
% Output arguments
%
%     fmask      0: clear land
%                1: clear water
%                2: cloud shadow
%                3: snow
%                4: cloud
%                255: filled (outside)
%
% Examples
%
%     clr_pct = autoFmask('cloud',0, 'shadow', 0) will produce the mask without buffers.
%     clr_pct = autoFmask('p',20) forces cloud probablity thershold as 20.
%     clr_pct = autoFmask('e',500) forces erosion radius for Potential False Positive Cloud as 500 meters to remove the large commission errors.
%
%        
% Author:  Shi Qiu (shi.qiu@uconn.edu) and Zhe Zhu (zhe@uconn.edu)
% Last Date: Jan. 16, 2025
% Copyright @ GERS Lab, UCONN.

    warning('off','all'); % do not show warning information
    tic
    fmask_soft_name='Fmask 4.7';
    fprintf('%s start ...\n',fmask_soft_name);
    path_data=pwd;
    
    %% get parameters from inputs
    p = inputParser;
    p.FunctionName = 'FmaskParas';
    % optional
    % default values.
    addParameter(p,'cloud',3);
    addParameter(p,'shadow',3);
    addParameter(p,'snow',0);
   
    %% read info from .xml.
    [sensor,~,InputFile,main_meta] = LoadSensorType(path_data);
    if isempty(sensor)
        error('%s works only for Landsats 4-7, Landsat 8, and Sentinel 2 images.\n',fmask_soft_name);
    end
    
    default_paras = FmaskParameters(sensor);
    tpw = default_paras.ThinWeight;
    addParameter(p,'d',default_paras.PFPCLayerExtensinRadius);
    addParameter(p,'e',default_paras.PFPCErosionRadius);
    addParameter(p,'p',default_paras.CloudProbabilityThershold);
    addParameter(p,'resolution',default_paras.OutputResolution);
    
    addParameter(p,'sw',default_paras.ShadowWater);
    
    % user's path for DEM
    addParameter(p,'udem','');

    % user's path for the auxiliaray data
    addParameter(p,'auxi','');
    
    % request user's input
    parse(p,varargin{:});
    resolution=p.Results.resolution;
    cldpix=p.Results.cloud;
    sdpix=p.Results.shadow;
    snpix=p.Results.snow;
    expdpix = round(p.Results.d/resolution);
    erdpix=round(p.Results.e/resolution);
    cldprob=p.Results.p;
    isShadowater = p.Results.sw;

    % users can use the local dem.
    userdem = p.Results.udem;
    % input the folder of auxiliaray data
    pathauxi = p.Results.auxi;
    clear p;
    
    fprintf('Cloud/cloud shadow/snow dilated by %d/%d/%d pixels.\n',cldpix,sdpix,snpix);
    fprintf('Cloud probability threshold of %.2f%%.\n',cldprob);
    
    fprintf('Load or calculate TOA reflectances.\n');
    
    %% load data
    [data_meta,data_toabt,angles_view,trgt] = LoadData(path_data,sensor,InputFile,main_meta);
    clear InputFile norMTL;
        
    if isempty(userdem)
        % default DEM
        [dem,slope,aspect,water_occur] = LoadAuxiData(fullfile(path_data,data_meta.Output),data_meta.Name,data_meta.BBox,trgt,false, 'auxi', pathauxi); % true false
    else
        [dem,slope,aspect,water_occur] = LoadAuxiData(fullfile(path_data,data_meta.Output),data_meta.Name,data_meta.BBox,trgt,false, 'userdem',userdem, 'auxi', pathauxi); % true false
    end

    fprintf('Detect potential clouds, cloud shadows, snow, and water.\n');
    
    %% public data
    mask=ObservMask(data_toabt.BandBlue);
    
    % a pixel's DEM can be set as the lowest value derived from the all workable pixels.
    if ~isempty(dem)
        dem_nan=isnan(dem);
        dem(dem_nan)=double(prctile(dem(~dem_nan&mask),0.001)); % exclude DEM errors.
        clear dem_nan;
    end
    
    % NDVI NDSI NDBI
    ndvi = NDVI(data_toabt.BandRed, data_toabt.BandNIR);
    ndsi = NDSI(data_toabt.BandGreen, data_toabt.BandSWIR1);
    cdi = CDI(data_toabt.BandVRE3,data_toabt.BandNIR8,data_toabt.BandNIR);%  band 7, 8, AND 8a

    data_toabt.BandVRE3 = [];
    data_toabt.BandNIR8 = [];

    % Statured Visible Bands
    satu_Bv = Saturate(data_toabt.SatuBlue, data_toabt.SatuGreen, data_toabt.SatuRed);
    data_toabt.SatuBlue = [];
    
    %% select potential cloud pixels (PCPs)
     % inputs: BandSWIR2 BandBT BandBlue BandGreen BandRed BandNIR BandSWIR1
    % BandCirrus
%     [idplcd,BandCirrusNormal,whiteness,HOT] = DetectPotentialPixels(mask,data_toabt,dem,ndvi,ndsi,satu_Bv);
   
    %% detect snow
    psnow = DetectSnow(data_meta.Dim, data_toabt.BandGreen, data_toabt.BandNIR, data_toabt.BandBT, ndsi);
    
    %% detect water
    [water, waterAll] = DetectWater(data_meta.Dim, mask, data_toabt.BandNIR, ndvi, psnow, slope, water_occur);
    clear water_occur;
    
    [idplcd,BandCirrusNormal,whiteness,HOT] = DetectPotentialPixels(mask,data_toabt,dem,ndvi,ndsi,satu_Bv);

    data_toabt.BandBlue = [];
    data_toabt.BandRed = [];
    data_toabt.BandSWIR2 = [];
    clear satu_Bv;
    data_toabt.BandCirrus = BandCirrusNormal; %refresh Cirrus band.
    clear BandCirrusNormal;

    %% select pure snow/ice pixels.
    abs_snow = DetectAbsSnow(data_toabt.BandGreen,data_toabt.SatuGreen,ndsi,psnow,data_meta.Resolution(1));

    if ~isnan(abs_snow)
        idplcd(abs_snow==1)=0; clear abs_snow; % remove pure snow/ice pixels from all PCPs.
    end

    %% detect potential cloud 
    ndbi = NDBI(data_toabt.BandNIR, data_toabt.BandSWIR1);
    
    % inputs: BandCirrus BandBT BandSWIR1 SatuGreen SatuRed
    [sum_clr,pcloud_all,idlnd,t_templ,t_temph]=DetectPotentialCloud(data_meta,mask,water,data_toabt, dem, ndvi,ndsi,ndbi,idplcd,whiteness,HOT,tpw,cldprob);
    clear ndsi idplcd whiteness HOT tpw cldprob;
    
    data_toabt.SatuGreen = [];
    data_toabt.SatuRed = [];
    data_toabt.BandCirrus = [];
    
    %% detect potential flase positive cloud layer, including urban, coastline, and snow/ice.
    pfpl = DetectPotentialFalsePositivePixels(mask, psnow, slope, ndbi, ndvi, data_toabt.BandBT,cdi, water,data_meta.Resolution(1));

    clear ndbi ndvi;
    
     % buffer the potential false positive cloud layer.
    if expdpix>0
        PFPCEs=strel('square',2*expdpix+1);
        pfpl=imdilate(pfpl,PFPCEs);
        clear PFPCEs;
    end
    clear expdpix;
    
    %% remove most of commission errors from urban, bright rock, and coastline.
    pcloud = ErodeCommissons(data_meta,pcloud_all,pfpl,water,cdi,erdpix);
    clear cdi;
    
    %% detect cloud shadow
    cs_final = zeros(data_meta.Dim,'uint8');  % final masks, including cloud, cloud shadow, snow, and water.
    cs_final(water==1)=1; %water is fistly stacked because of its always lowest prioty.

    % note that 0.1% Landsat obersavtion is about 40,000 pixels, which will be used in the next statistic analyses.
    % when potential cloud cover less than 0.1%, directly screen all PCPs out.
    if sum_clr <= 40000
        fprintf('No clear pixel in this image (clear-sky pixels = %.0f%)\n',sum_clr);
        pcloud=pcloud>0;
        pshadow=~pcloud;
        clear data_toabt;
    else
        fprintf('Match cloud shadows with clouds.\n');
        % detect potential cloud shadow
        pshadow = DetectPotentialCloudShadow(data_meta, data_toabt.BandNIR,data_toabt.BandSWIR1,idlnd,mask,...
        slope,aspect);
    
        data_toabt.BandNIR = [];
        data_toabt.BandSWIR1 = [];
        
        data_bt_c=data_toabt.BandBT;
        clear data_toabt;
        % match cloud shadow, and return clouds and cloud shadows.
        [ ~,pcloud, pshadow] = MatchCloudShadow(...
        mask,pcloud,pshadow,isShadowater,waterAll, dem ,data_bt_c,t_templ,t_temph,data_meta,sum_clr,14,angles_view);
   
        % make buffer for final masks.
        % the called cloud indicate those clouds are have highest piroity.
        % This is final cloud!
        [pcloud,pshadow,psnow] = BufferMasks(pcloud,cldpix,pshadow,sdpix,psnow,snpix);
    end
    %% stack results together. 
    % step 1 snow or unknow
    cs_final(psnow==1)=3; % snow
    % step 2 shadow above snow and everyting
    cs_final(pshadow==1)=2; %shadow
    % step 3 cloud above all
    cs_final(pcloud==1)=4; % cloud
    % mask out no data.
    cs_final(mask==0)=255; % mask
    
    % clear pixels percentage
    clr_pct=100*(1-sum(pcloud(:))/sum(mask(:)));

    %% output as geotiff.
    trgt.Z=cs_final;
    fmask_name=[data_meta.Name,'_Fmask4'];
    trgt.name=fmask_name;
 
    fmask_output=fullfile(path_data,data_meta.Output,[fmask_name,'.tif']);
    GRIDobj2geotiff(trgt,fmask_output);
	time=toc;
    time=time/60;
    fprintf('%s finished (%.2f minutes)\nfor %s with %.2f%% clear pixels\n\n',...
        fmask_soft_name,time,data_meta.Name,clr_pct);
end
