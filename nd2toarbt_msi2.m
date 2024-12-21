function [im_th,TOAref,trgt,ijdim_ref,bbox,ul,zen,azi,zc,Angles2,B1Satu,B2Satu,B3Satu,resolu]=nd2toarbt_msi2(im, band_offsets, quantif)
% read TOA refs function derived from Fmask 4.4 for Sentinel 2.
% Revisions:
% Designed for Sentinel-2 Baseline 4.00 (Shi /26/12/2021)
% Use REF vs. DN instead of RAD vs. DN (Zhe 06/20/2013)
% Combined the Earth-Sun distance table into the function (Zhe 04/09/2013)
% Process Landsat 8 DN values (Zhe 04/04/2013)
% Proces the new metadata for Landsat TM/ETM+ images (Zhe 09/28/2012)
% Fixed bugs caused by earth-sun distance table (Zhe 01/15/2012)
%
% [im_th,TOAref,ijdim_ref,ul,zen,azi,zc,B1Satu,B2Satu,B3Satu,resolu]=nd2toarbt(filename)
% Where:
% Inputs:
% im= MSI filename structure including:
% - im.Dmain (root directory of the SAFE directory)
% - im.DataStrip directory (without ".SAFE")
% - im.Granule directory;
% im.Dmain = '/home/bernie/MSIdata/';
% im.DataStrip = 'S2A_OPER_PRD_MSIL1C_PDMC_20151229T234852_R139_V20151229T144823_20151229T144823';
% im.Granule  = 'S2A_OPER_MSI_L1C_TL_SGS__20151229T201123_A002710_T20QPD_N02.01';
% Outputs:
% 1) im_th = Brightness Temperature (BT)
% 2) TOAref = Top Of Atmoshpere (TOA) reflectance
% 3) ijdim = [nrows,ncols]; % dimension of optical bands
% 4) ul = [upperleft_mapx upperleft_mapy];
% 5) zen = solar zenith angle (degrees);
% 6) azi = solar azimuth angle (degrees);
% 7) zc = Zone Number
% 8,9,10) Saturation (true) in the Visible bands
% 11) resolution of Fmask results
% eg.

    FilesInfo.DirIn = im.Dmain;
    FilesInfo.DataStrip =im.DataStrip ;
    FilesInfo.Granule=im.Granule;
    
    FilesInfo.InspireXML=im.InspireXML;
    clear im;

    % obtain them from INSPIRE.xml
    bbox = ReadS2InspireXML(FilesInfo.InspireXML);

    %% Metadata read ReadSunViewGeometryMSI(DataStrip,Granule,BandSel,PsizeOut,Dmain)
     [MSIinfo,Angles]  = ReadSunViewGeometryMSIBaseline04 (FilesInfo.DataStrip,FilesInfo.Granule,4,10,FilesInfo.DirIn);

    Angles2.VAA  = Angles.VAA_B04 ;
    Angles2.VZA  = Angles.VZA_B04 ;
    clear Angles;

    %% output resolution of Fmask 20meters for Sentinel 2 images
    resolu = [20 20] ;
    %%
    ijdim_ref = (MSIinfo.GeoInfo.Size.R10) *10 ./ resolu ;

    ul  = [MSIinfo.GeoInfo.Xstart.R10 MSIinfo.GeoInfo.Ystart.R10] + [resolu(1)/2 0-resolu(2)/2]; % the center of the top-left pixel.
    zen = MSIinfo.Angles.Mean.SZA ;
    azi = MSIinfo.Angles.Mean.SAA ;

    zc_num=MSIinfo.GeoInfo.UTM_zone(1:end-1);
    
    % convert UTM zone to code by refering the bellow rule.
    % ref. http://geotiff.maptools.org/spec/geotiff6.html#6.3.3.1
    zc_ns=MSIinfo.GeoInfo.UTM_zone(end);
    clear MSIinfo;
    if zc_ns=='N'
        zc = abs(str2double(zc_num));
        if zc>10
            geokey = ['326',zc_num]; 
        else
            geokey = ['3260',zc_num];
        end
        geokey=str2double(geokey);
    elseif zc_ns=='S'
        zc = abs(str2double(zc_num));
        if zc>10
            geokey = ['327',zc_num];
        else
            geokey = ['3270',zc_num];
        end
        geokey=str2double(geokey);
    end


    %% open MSI data
    BandCorr = {'02','03','04','8A','11','12','10','07','08'};
    BandCorrId = [1, 2, 3, 8, 11, 12, 10, 6, 7];
    Ratio = [10 10 10 20 20 20 60 20 10] /(resolu(1)); % resample all bands to the same resolu
    Dmain =  fullfile(FilesInfo.DirIn, [FilesInfo.DataStrip '.SAFE'],'GRANULE', FilesInfo.Granule, 'IMG_DATA');
    id_missing=zeros(ijdim_ref,'uint8');
    for iB=1:length(BandCorr)        
        % support all versions for Sentinel 2 images.
        tempfn = dir(fullfile(Dmain, [FilesInfo.Granule(1) '*B' BandCorr{iB} '.jp2']));
        if isempty(tempfn)
            tempfn = dir(fullfile(Dmain, [FilesInfo.Granule(5) '*B' BandCorr{iB} '.jp2']));
            if isempty(tempfn)
                tempfn = dir(fullfile(Dmain, ['*B' BandCorr{iB} '.jp2']));
            end
        end
        dum=imread(fullfile(Dmain, tempfn(1).name));
        mask_filled =  dum == 0;
        dum = 10000.*(single(dum) + band_offsets(BandCorrId(iB) == band_offsets(:, 1), 2))./quantif; % OFFset
        clear tempfn;
        
        dum(mask_filled)=NaN;
        clear mask_filled;
        if Ratio(iB)<1 % box-agg pixel
            TOAref(:,:,iB) = imresize(dum,Ratio(iB),'box');
        elseif Ratio(iB)>1 % split pixel
            TOAref(:,:,iB) = imresize(dum,Ratio(iB),'nearest');
        elseif Ratio(iB)==1
            TOAref(:,:,iB) = dum;
        end
        clear dum;
        
% %         if isequal(BandCorr{iB},'08')
% %             % Gaussfilt for band 8 to better scall up.
% %             TOAref(:,:,iB) = imgaussfilt(TOAref(:,:,iB));
% %         end
        
        % only processing pixesl where all bands have values (id_mssing)
        id_missing=id_missing|isnan(TOAref(:,:,iB));
    end
    clear Dmain FilesInfo BandCorr Ratio iB;
%     trgt=CreateTargetGRIDobj(info_jp2.BoundingBox,resolu,ul,ijdim_ref,zc_num,zc_ns,geokey);
    trgt=CreateTargetGRIDobj(resolu,ul,ijdim_ref,zc_num,zc_ns,geokey);

    trgt.Z=[];
    
%     trgt_test = GRIDobj('E:\New validation dataset\New accuracy assessment samples for Fmask 4_0\Sentinel2\S2A_MSIL1C_20171011T123151_N0205_R023_T17CNL_20171011T123149.SAFE\GRANULE\L1C_T17CNL_A012032_20171011T123149\Samples\assist data\Samples.tif');
    
    %%
    %%%%% WARNING - what is the MSI fill value? what is the saturation flag ? 65535
%     TOAref(isnan(TOAref))=-9999;
    TOAref(id_missing)=-9999;
    clear id_missing;
    %%%%%%%%%%%% Not used but same format
    im_th=-9999;

    % B1Satu = zeros([size(TOAref,1) size(TOAref,2)],'uint8')==1;
    % B2Satu = zeros([size(TOAref,1) size(TOAref,2)],'uint8')==1;
    % B3Satu = zeros([size(TOAref,1) size(TOAref,2)],'uint8')==1;

    % find pixels that are saturated in the visible bands
    % SATURATED VALUE: 65535 set by Shi 10/18/2017
    B1Satu=TOAref(:,:,1)==65535;
    B2Satu=TOAref(:,:,2)==65535;
    B3Satu=TOAref(:,:,3)==65535;
end

function trgt=CreateTargetGRIDobj(resolu,ul,ijdim_ref,zc_num,zc_ns,geokey)

    trgt = GRIDobj([]);
    trgt.name='target';
    trgt.cellsize=resolu(1);

    ul_tmp=ul - [resolu(1)/2 0-resolu(2)/2]; %back to limits.

    % location of the 'center' of the first (1,1) pixel in the image.
    % trgt.refmat=makerefmat(ul(1),ul(2),resolu(1),0-resolu(2));
    trgt.refmat = maprefcells([ul(2) ul(2) + resolu(2)], [ul(1) ul(1) + resolu(1)], resolu);
    clear ul;
    trgt.size=ijdim_ref;

    % boundary.
    xWorldLimits=[ul_tmp(1),ul_tmp(1)+resolu(1)*(ijdim_ref(1))];
    yWorldLimits=[ul_tmp(2)-resolu(2)*(ijdim_ref(2)),ul_tmp(2)];
    clear ul_tmp resolu;

        spatialRef = maprefcells(xWorldLimits,yWorldLimits,ijdim_ref,...
            'ColumnsStartFrom','north');
    clear xWorldLimits yWorldLimits ijdim_ref;
    trgt.georef.SpatialRef=spatialRef;
    clear spatialRef;
% %     http://www.gdal.org/frmt_sentinel2.html
    trgt.georef.GeoKeyDirectoryTag.GTModelTypeGeoKey = 1;
    trgt.georef.GeoKeyDirectoryTag.GTRasterTypeGeoKey = 1;
    trgt.georef.GeoKeyDirectoryTag.GTCitationGeoKey = ['WGS 84 / UTM zone ',zc_num,zc_ns];
    trgt.georef.GeoKeyDirectoryTag.ProjectedCSTypeGeoKey = geokey; clear geokey;
    trgt.georef.GeoKeyDirectoryTag.PCSCitationGeoKey = ['WGS 84 / UTM zone ',zc_num,zc_ns]; % same

%     ellipsoid = utmgeoid([zc_num,zc_ns]);

    % this can be used as follows because the Sentinel 2 titles's projection
    % will not be changable.
    % WGS84 UTM: A UTM zone is a 6?° segment of the Earth.
    E = wgs84Ellipsoid('meters');
    utmstruct = defaultm('tranmerc');
    utmstruct.geoid =  [E.SemimajorAxis,E.Eccentricity];
    clear E;
    % UTM false East (m)  
    % the central meridian is assigned 500,000 meters in each zone.
    utmstruct.falseeasting=500000;
    % UTM false North (m)
    % If you are in the northern hemisphere, the equator has a northing 
    % value of 0 meters. In the southern hemisphere, the equator starts
    % at 10,000,000 meters. This is because all values south of the 
    % equator will be positive.
    if zc_ns=='N'
        utmstruct.falsenorthing=0; 
    else % S
        utmstruct.falsenorthing=10000000;% 10,000,000 
    end
    clear zc_ns;
    % UTM scale factor
    utmstruct.scalefactor=0.9996; 
    % each UTM zone is 6 degrees. a total of 60 zones.
    origin=(str2double(zc_num)-31).*6+3;
    clear zc_num;
    utmstruct.origin=[0,origin,0];
    clear origin;
    utmstruct = defaultm(utmstruct);    
    trgt.georef.mstruct=utmstruct;
    clear utmstruct;
end
