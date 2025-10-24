"""define default predictors"""
l8_dataset = [
    "L8BIOME",
    "L8SPARCS",
    "L895CLOUD",
]  # training datasets

s2_dataset = [
    "S2ALCD",
    "S2IRIS",
    "S2WHUCDPLUS",
    "S2FMASK",
    "S2CLOUDSEN12PLUS",
]  # training datasets

l8_predictor_full = [
    "coastal",
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "tirs1",
    "tirs2",
    "cirrus",
    "hot",
    "whiteness",
    "ndvi",
    "ndsi",
    "ndbi",
    "sfdi",
    "var_nir",
    "dem",
    "swo",
]  # predictors for full model

l8_predictor_cloud_phy = [
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "tirs1",
    "cirrus",
    "hot",
    "whiteness",
    "ndvi",
    "ndsi",
    "ndbi",
    "dem",
    "swo",
]

l8_predictor_cloud_pixel = [
    "coastal",
    "tirs1",
    "tirs2",
    "cirrus",
    "ndvi",
    "ndbi",
    "dem",
]  # top ten predictors for cloud pixel model

l8_predictor_cloud_cnn = [
    "coastal",
    "blue",
    "green",
    "red",
    "nir",
    "whiteness",
    "swo",
]  # predictors for cloud CNN model

l8_predictor_shadow_cnn = [
    "nir",
    "swir1",
    "dem"
]  # predictors for shadow CNN model

l7_predictor_full = [
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "tirs1",
    "hot",
    "whiteness",
    "ndvi",
    "ndsi",
    "ndbi",
    "sfdi",
    "var_nir",
    "dem",
    "swo",
]  # predictors for full model

l7_predictor_cloud_phy = [
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "tirs1",
    "hot",
    "whiteness",
    "ndvi",
    "ndsi",
    "ndbi",
    "dem",
    "swo",
]

l7_predictor_cloud_pixel = [
    "blue",
    "green",
    "red",
    "swir1",
    "swir2",
    "tirs1",
    "ndvi",
    "ndsi",
    "ndbi",
    "dem",
]  # predictors for cloud pixel model

l7_predictor_cloud_cnn = [
    "blue",
    "red",
    "nir",
    "whiteness",
    "swo",
]  # predictors for cloud CNN model

l7_predictor_shadow_cnn = [
    "nir",
    "swir1",
    "dem"
]  # predictors for shadow CNN model

s2_predictor_full = [
    "coastal",
    "blue",
    "green",
    "red",
    "vre1",
    "vre2",
    "vre3",
    "wnir",
    "nir",
    "wv",
    "swir1",
    "swir2",
    "cirrus",
    "hot",
    "whiteness",
    "ndvi",
    "ndsi",
    "ndbi",
    "cdi",
    "sfdi",
    "var_nir",
    "dem",
    "swo",
]  # predictors for full model

s2_predictor_cloud_phy = [
    "blue",
    "green",
    "red",
    "vre3",
    "wnir",
    "nir",
    "swir1",
    "swir2",
    "cirrus",
    "hot",
    "whiteness",
    "ndvi",
    "ndsi",
    "ndbi",
    "cdi",
    "dem",
    "swo",
]  # predictors for full model

s2_predictor_cloud_pixel = [
    "coastal",
    "vre2",
    "wv",
    "swir1",
    "cirrus",
    "hot",
    "ndvi",
    "ndsi",
    "sfdi",
    "var_nir",
    "cdi",
    "swo",
]  # predictors for cloud pixel model

s2_predictor_cloud_cnn = [
    "coastal",
    "green",
    "swir1",
    "cirrus",
    "ndsi",
    "sfdi",
    "dem",
]  # predictors for cloud CNN model, similar accuracy, but fewer predictors

s2_predictor_cloud_cnn_10layer = [
    "coastal",
    "blue",
    "green",
    "red",
    "vre2",
    "vre3",
    "nir",
    "swir1",
    "whiteness",
    "ndsi",
]  # predictors for cloud CNN model, ten layers were selected

s2_predictor_shadow_cnn = [
    "nir",
    "swir1",
    "dem",
]  # predictors for shadow CNN model

# End of the script
