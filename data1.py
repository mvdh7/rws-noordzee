#%% Import packages ====================================================================
#   ___                _
#  (  _`\             (_ )
#  | |_) ) _ __   _    | |    _      __   _   _    __
#  | ,__/'( '__)/'_`\  | |  /'_`\  /'_ `\( ) ( ) /'__`\
#  | |    | |  ( (_) ) | | ( (_) )( (_) || (_) |(  ___/
#  (_)    (_)  `\___/'(___)`\___/'`\__  |`\___/'`\____)
#                                 ( )_) |
#                                  \___/'

import copy, pickle, re, datetime
import numpy as np, pandas as pd, matplotlib as mpl, xarray as xr
from scipy.stats import linregress
from matplotlib import pyplot as plt, dates as mdates, units as munits
import PyCO2SYS as pyco2

# # Use environment rws_dev
# from sys import path

# for extra in ["/home/matthew/github/koolstof"]:  # , "C:/Users/mphum/GitHub/calkulate"]:
#     if extra not in path:
#         path.append(extra)

import koolstof as ks, calkulate as calk

#%% Set settings =======================================================================
#   ___           _    _
#  (  _`\        ( )_ ( )_  _
#  | (_(_)   __  | ,_)| ,_)(_)  ___     __    ___
#  `\__ \  /'__`\| |  | |  | |/' _ `\ /'_ `\/',__)
#  ( )_) |(  ___/| |_ | |_ | || ( ) |( (_) |\__, \
#  `\____)`\____)`\__)`\__)(_)(_) (_)`\__  |(____/
#                                    ( )_) |
#                                     \___/'

# ~~~ General settings: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Whether or not to do certain things
draw_figures = True  # draw figures?
save_figures = False  # save figures to file? (if draw_figures = True)

# PyCO2SYS options, also used by Calkulate
opt_k_carbonic = 10
opt_total_borate = 1
opt_k_bisulfate = 1
opt_k_fluoride = 2

# Calkulate options
temperature_override = None  # set to None to ignore; ~23 makes the pH work very well

# Poison corrections for DIC and alkalinity
sample_volume = 250e-3  # sample bottles in ml
poison_volume = 250e-6  # added HgCl2 in ml
# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~

# Matplotlib date epoch (default changed in v3.3...)
mpl.rcParams["date.epoch"] = "1970-01-01T00:00:00"

# Matplotlib universal settings
mpl.rcParams["font.family"] = "Open Sans"
mpl.rcParams["font.size"] = mpl.rcParams["axes.titlesize"] = 10
converter = mdates.ConciseDateConverter()
munits.registry[np.datetime64] = converter
munits.registry[datetime.date] = converter
munits.registry[datetime.datetime] = converter

#%% Import VINDTA files ================================================================
#   _   _  _  _   _  ___   _____  _____       ___    _
#  ( ) ( )(_)( ) ( )(  _`\(_   _)(  _  )    /'___)_ (_ )
#  | | | || || `\| || | ) | | |  | (_) |   | (__ (_) | |    __    ___
#  | | | || || , ` || | | ) | |  |  _  |   | ,__)| | | |  /'__`\/',__)
#  | \_/ || || |`\ || |_) | | |  | | | |   | |   | | | | (  ___/\__, \
#  `\___/'(_)(_) (_)(____/' (_)  (_) (_)   (_)   (_)(___)`\____)(____/

# ~~~ File import settings: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
vindta_path = "data/vindta15_data/"  # path to the logfile and .dbs files
logfile_fname = "logfile_20210625.bak"  # the logfile's filename
dbs_fnames = (  # list of .dbs filenames
    ["2018/2018_{}_RWS_CO2.dbs".format(month) for month in ["June", "Aug", "Nov"]]
    + ["2019/2019_{}_RWS_CO2.dbs".format(month) for month in ["02", "08", "11"]]
    + [
        "2020/2020_07_RWS_CO2.dbs",
        "2020/2020_11_RWS_CO2.dbs",
        "2021/2021_05_RWS_CO2.dbs",
        "2021/2021_06_RWS_CO2_Batch10.dbs",
    ]
)
methods = [  # list of VINDTA methods that are considered as measurements
    "3C standard",
    "3C standardRWS",
]
# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~

# Import VINDTA logfile and .dbs file(s)
logfile = ks.read_logfile(vindta_path + logfile_fname, methods=methods)
# "Logfile line 31691: bottle name not found!" warning can be ignored
dbs = ks.concat(
    [ks.read_dbs(vindta_path + fname) for fname in dbs_fnames], logfile=logfile
)

# ~~~ QC the logfile: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
logfile.loc[
    (logfile.bottle == "NOORDWK2_2019006695")
    & (mdates.date2num(logfile.analysis_datetime) > 18121),
    "bottle",
] = "NOORDWK2_2019006695B"
# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~

# Get index for logfile rows corresponding to the dbs table
dbs.get_logfile_index()

#%% Import metadata ====================================================================
#   ___           _                         _               _         _
#  (  _`\        ( )_                      ( )_            ( )       ( )_
#  | ( (_)   __  | ,_)     ___ ___     __  | ,_)   _ _    _| |   _ _ | ,_)   _ _
#  | |___  /'__`\| |     /' _ ` _ `\ /'__`\| |   /'_` ) /'_` | /'_` )| |   /'_` )
#  | (_, )(  ___/| |_    | ( ) ( ) |(  ___/| |_ ( (_| |( (_| |( (_| || |_ ( (_| |
#  (____/'`\____)`\__)   (_) (_) (_)`\____)`\__)`\__,_)`\__,_)`\__,_)`\__)`\__,_)

# ~~~ Import data and stations tables: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
with open("pickles/data0_stations_groups_v10.pkl", "rb") as f:
    data, stations, groups = pickle.load(f)

# # Rename columns to avoid clashes
# data.rename(mapper={"dic": "dic_original"}, axis=1, inplace=True)

# Define dict of columns of data (keys) to transfer to dbs (values) below
data2dbs = {
    "salinity": "salinity",
    "silicate_vol": "silicate_vol",
    "phosphate_vol": "phosphate_vol",
    "sulfate_vol": "sulfate_vol",
    "ammonia_vol": "ammonia_vol",
}
# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~


def get_data_index(dbs_row, data):
    """Get index of data in dbs to transfer metadata across for processing."""
    data_iloc = np.nan  # stays this way if no match found
    # dbs["bottle"] is exactly correct:
    if dbs_row.bottle in data.station_bottleid.values:
        data_iloc = np.where(dbs_row.bottle == data.station_bottleid.values)[0]
    # dbs["bottle"] ends with "R" or "B":
    elif (
        dbs_row.bottle.endswith("R") or dbs_row.bottle.endswith("B")
    ) and dbs_row.bottle[:-1] in data.station_bottleid.values:
        data_iloc = np.where(dbs_row.bottle[:-1] == data.station_bottleid.values)[0]
    # dbs["bottle"] starts with "TERSGL" instead of "TERSLG":
    elif (dbs_row.bottle.startswith("TERSGL")) and dbs_row.bottle.replace(
        "TERSGL", "TERSLG"
    ) in data.station_bottleid.values:
        data_iloc = np.where(
            dbs_row.bottle.replace("TERSGL", "TERSLG") == data.station_bottleid.values
        )[0]
    # dbs["bottle"] has an "R" just before the underscore:
    elif (
        "R_" in dbs_row.bottle
        and dbs_row.bottle.replace("R_", "_") in data.station_bottleid.values
    ):
        data_iloc = np.where(
            dbs_row.bottle.replace("R_", "_") == data.station_bottleid.values
        )[0]
    # dbs["bottle"] has an "B" just before the underscore:
    elif (
        "B_" in dbs_row.bottle
        and dbs_row.bottle.replace("B_", "_") in data.station_bottleid.values
    ):
        data_iloc = np.where(
            dbs_row.bottle.replace("B_", "_") == data.station_bottleid.values
        )[0]
    # dbs["bottle"] has an "b" just before the underscore:
    elif (
        "b_" in dbs_row.bottle
        and dbs_row.bottle.replace("b_", "_") in data.station_bottleid.values
    ):
        data_iloc = np.where(
            dbs_row.bottle.replace("b_", "_") == data.station_bottleid.values
        )[0]
    # Finally, assign data_index:
    if np.isnan(data_iloc):
        data_index = np.nan
    else:
        assert (
            len(data_iloc) == 1
        ), "Found more than one bottle ID match for {}!".format(dbs_row.bottle)
        data_index = data.index[data_iloc[0]]
        data.loc[data_index, "in_dbs"] = True
    return data_index


# Get indices of data in dbs
data["in_dbs"] = False
dbs["data_index"] = dbs.apply(get_data_index, args=[data], axis=1)


def fix_data2dbs_typo(l_dbs, l_data):
    """Set indices manually to deal with typos in dbs["bottle"]."""
    assert l_dbs.sum() == 1
    assert l_data.sum() == 1
    dbs.loc[l_dbs, "data_index"] = data.index[l_data][0]
    data.loc[l_data, "in_dbs"] = True


# ~~~ Set indices to deal with typos: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fix_data2dbs_typo(
    (dbs.bottle == "SHOUWN10_2018005608") & (dbs.dic_cell_id == "C_Jun07-18_0706"),
    data.station_bottleid == "SCHOUWN10_2018005608",
)
fix_data2dbs_typo(
    (dbs.bottle == "NOORDWK20A_2018005672") & (dbs.dic_cell_id == "C_Aug13-18_1008"),
    data.station_bottleid == "NOORDWK20_2018005672",
)
fix_data2dbs_typo(
    (dbs.bottle == "TERSLG_2018005727") & (dbs.dic_cell_id == "C_May02-19_0705"),
    data.station_bottleid == "TERSLG100_2018005727",
)
fix_data2dbs_typo(
    (dbs.bottle == "TERSLG135_2018005725737") & (dbs.dic_cell_id == "C_Nov21-18_0811"),
    data.station_bottleid == "TERSLG135_2018005737",
)
fix_data2dbs_typo(
    (dbs.bottle == "ROTTMPT70_209008007") & (dbs.dic_cell_id == "C_Nov25-19_0911"),
    data.station_bottleid == "ROTTMPT70_2019008007",
)
fix_data2dbs_typo(
    (dbs.bottle == "GOERE6_2018005623") & (dbs.dic_cell_id == "C_Jun08-18_0706"),
    data.station_bottleid == "GOERE6_2018005635",
)
fix_data2dbs_typo(
    (dbs.bottle == "ROTTMPT50_2018005766") & (dbs.dic_cell_id == "C_Aug14-18_0708"),
    data.station_bottleid == "ROTTMPT50_2018008766",
)
fix_data2dbs_typo(
    (dbs.bottle == "ROTTMPT70_2018005773") & (dbs.dic_cell_id == "C_Aug14-18_0708"),
    data.station_bottleid == "ROTTMPT70_2018008773",
)
fix_data2dbs_typo(
    (dbs.bottle == "ROTTMPT50_2018005767") & (dbs.dic_cell_id == "C_Aug14-18_0708"),
    data.station_bottleid == "ROTTMPT50_2018008767",
)
fix_data2dbs_typo(
    (dbs.bottle == "ROTTMPT70_2018005774") & (dbs.dic_cell_id == "C_Aug14-18_1308"),
    data.station_bottleid == "ROTTMPT70_2018008774",
)
fix_data2dbs_typo(
    (dbs.bottle == "RTTMPT50_2018005769") & (dbs.dic_cell_id == "C_Aug14-18_1308"),
    data.station_bottleid == "ROTTMPT50_2018008769",
)
fix_data2dbs_typo(
    (dbs.bottle == "RTTMPT70_2018005776") & (dbs.dic_cell_id == "C_Aug14-18_1308"),
    data.station_bottleid == "ROTTMPT70_2018008776",
)
fix_data2dbs_typo(
    (dbs.bottle == "ROTTMPT50_2018005770") & (dbs.dic_cell_id == "C_Aug15-18_0708"),
    data.station_bottleid == "ROTTMPT50_2018008770",
)
fix_data2dbs_typo(
    (dbs.bottle == "ROTTMPT70_2018005777") & (dbs.dic_cell_id == "C_Aug15-18_0708"),
    data.station_bottleid == "ROTTMPT70_2018008777",
)
fix_data2dbs_typo(
    (dbs.bottle == "ROTTMPT70_2018005775") & (dbs.dic_cell_id == "C_Aug15-18_0708"),
    data.station_bottleid == "ROTTMPT70_2018008775",
)
fix_data2dbs_typo(
    (dbs.bottle == "ROTTMPT70_2018005580") & (dbs.dic_cell_id == "C_Nov20-18_0811"),
    data.station_bottleid == "ROTTMPT70_2018008778",
)
fix_data2dbs_typo(
    (dbs.bottle == "NOORDWK10B_2018005567") & (dbs.dic_cell_id == "C_May02-19_0705"),
    data.station_bottleid == "NOORDWK10_2018005667",
)
fix_data2dbs_typo(
    (dbs.bottle == "NOORDWK20") & (dbs.dic_cell_id == "C_Aug12-19_0708"),
    data.station_bottleid == "NOORDWK20_2019006697",
)  # but bad AT, use the one from the following day
fix_data2dbs_typo(
    (dbs.bottle == "TERSLG175_202005973") & (dbs.dic_cell_id == "C_Jul09-20_0707"),
    data.station_bottleid == "TERSLG175_2020005973",
)
fix_data2dbs_typo(
    (dbs.bottle == "TERSLG235_202005974") & (dbs.dic_cell_id == "C_Jul09-20_0707"),
    data.station_bottleid == "TERSLG235_2020005974",
)
fix_data2dbs_typo(
    (dbs.bottle == "ROTTMPT50_202005975") & (dbs.dic_cell_id == "C_Jul09-20_0707"),
    data.station_bottleid == "ROTTMPT50_2020005975",
)
fix_data2dbs_typo(
    (dbs.bottle == "ROTTMPT70_202005976") & (dbs.dic_cell_id == "C_Jul09-20_0707"),
    data.station_bottleid == "ROTTMPT70_2020005976",
)
fix_data2dbs_typo(
    (dbs.bottle == "NOORDWK10R_20200052154") & (dbs.dic_cell_id == "C_Jul09-20_0707"),
    data.station_bottleid == "NOORDWK10_2020005215",
)
fix_data2dbs_typo(
    (dbs.bottle == "NOORDWK10_2021005553A") & (dbs.dic_cell_id == "C_Jun23-21_0706"),
    data.station_bottleid == "NOORDWK10_2021005553",
)
# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~

# Transfer metadata from data to dbs with overwriting
for data_field, dbs_field in data2dbs.items():
    if dbs_field not in dbs.columns:
        dbs[dbs_field] = np.nan
    L_dbs = ~np.isnan(dbs.data_index)
    dbs.loc[L_dbs, dbs_field] = data.loc[dbs.loc[L_dbs].data_index][data_field].values

# Convert nutrients to molinity
nutrients_temperature = 23  # presumed - check! <-------------------------------- TO DO
dbs["density_nutrients"] = calk.density.seawater_1atm_MP81(
    temperature=nutrients_temperature, salinity=data.salinity
)
nutrients_dbs = ["silicate", "phosphate", "sulfate", "ammonia"]
for nutrient in nutrients_dbs:
    dbs[nutrient] = dbs[nutrient + "_vol"] / dbs.density_nutrients

# Convert all nutrients to molinity
data["density_nutrients"] = calk.density.seawater_1atm_MP81(
    temperature=nutrients_temperature, salinity=data.salinity
)
nutrients = [
    "doc",
    "poc",
    "nitrate",
    "nitrite",
    "ammonia",
    "din",
    "silicate",
    "phosphate",
    "sulfate",
    "chloride",
    "dn",
    "dp",
]
for nutrient in nutrients:
    data[nutrient] = data[nutrient + "_vol"] / data.density_nutrients

# Find non-matched measurements
dbs_ = dbs[
    np.isnan(dbs.data_index)
    & (dbs.station != 666)
    & ~dbs.bottle.str.upper().str.startswith("JUNK")
    & ~dbs.bottle.str.upper().str.startswith("CRM")
    & ~dbs.bottle.str.upper().str.startswith("TEST")
    & ~dbs.bottle.str.upper().str.startswith("NUTS")
    & ~dbs.bottle.str.upper().str.startswith("NUTL")
    & ~dbs.bottle.str.upper().str.startswith("WADD")
]

# Assign QUASIMEME salinity values
dbs.loc[dbs.bottle == "QOA001SW", "salinity"] = 34.1944
dbs.loc[dbs.bottle == "QOA002SW", "salinity"] = 35.4875
dbs.loc[dbs.bottle == "QOA003SW", "salinity"] = 8.1314

#%% Determine DIC coulometer blanks ====================================================
#   ___    _  ___       _      _                  _
#  (  _`\ (_)(  _`\    ( )    (_ )               ( )
#  | | ) || || ( (_)   | |_    | |    _ _   ___  | |/')   ___
#  | | | )| || |  _    | '_`\  | |  /'_` )/' _ `\| , <  /',__)
#  | |_) || || (_( )   | |_) ) | | ( (_| || ( ) || |\`\ \__, \
#  (____/'(_)(____/'   (_,__/'(___)`\__,_)(_) (_)(_) (_)(____/

# ~~~ Define when to start measuring blank: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
use_from = 6  # runtime (in minutes) after which blank is measured

# Correct errors in dic_cell_id
dbs["analysis_datenum"] = mdates.date2num(dbs.analysis_datetime)
dbs.loc[
    (dbs.dic_cell_id == "C_Jul10-20_0707") & (dbs.analysis_datenum > 18500),
    "dic_cell_id",
] = "C_Nov02-20_0911"
# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~

# Get sample-by-sample blanks
dbs.get_sample_blanks(use_from=use_from)

# ~~~ Define which blanks are reliable: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dbs["blank_good"] = dbs.blank_here < 1000
# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~

# Get session blanks and estimate for each sample, overwriting existing "blank" column
dbs.get_blank_corrections()

# Plot DIC coulometer increments and blanks
if draw_figures:
    dbs.plot_blanks(figure_path="figures/calibration/dic_blanks")

#%% Calibrate DIC measurements =========================================================
#   ___           _       _                  _              ___    _  ___
#  (  _`\        (_ )  _ ( )                ( )_           (  _`\ (_)(  _`\
#  | ( (_)   _ _  | | (_)| |_    _ __   _ _ | ,_)   __     | | ) || || ( (_)
#  | |  _  /'_` ) | | | || '_`\ ( '__)/'_` )| |   /'__`\   | | | )| || |  _
#  | (_( )( (_| | | | | || |_) )| |  ( (_| || |_ (  ___/   | |_) || || (_( )
#  (____/'`\__,_)(___)(_)(_,__/'(_)  `\__,_)`\__)`\____)   (____/'(_)(____/'

# ~~~ Identify CRMs and their batch numbers: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dbs["crm_batch"] = np.nan
dbs.loc[dbs.bottle == "JUNK#171_0419B", "station"] = 1  # fix typo
for i in dbs.index:
    if dbs.loc[i].station == 666:
        if dbs.loc[i].analysis_datenum < 18437:
            dbs.loc[i, "crm_batch"] = 171
        else:
            dbs.loc[i, "crm_batch"] = int(
                dbs.loc[i].bottle.split("CRM")[1].split("_")[0]
            )

# DIC analysis settings
dbs["temperature_analysis_dic"] = 23
# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~

# Get Dickson CRM values from their batch numbers
dbs["crm"] = ~np.isnan(dbs.crm_batch)
dickson_fields = [  # also get some fields needed for alkalinity
    "salinity",
    "dissolved_inorganic_carbon",
    "total_alkalinity",
    "silicate",
    "phosphate",
]
dickson_crms = ks.crm.dickson(dbs.crm_batch.values, dickson_fields)
for dfield in dickson_fields:
    if dfield == "dissolved_inorganic_carbon":
        field = "dic_certified"
    elif dfield == "total_alkalinity":
        field = "alkalinity_certified"
    else:
        field = dfield
    if field not in dbs.columns:
        dbs[field] = dickson_crms[dfield]
    else:
        dbs.loc[dbs.crm, field] = dickson_crms[dfield][dbs.crm]

# Calculate density of CRMs
dbs.get_standard_calibrations()

# ~~~ QC the CRM calibration factors: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dbs["k_dic_good"] = copy.deepcopy(dbs.crm)
dbs.loc[
    (dbs.k_dic_here < 0.011)
    & (dbs.dbs_fname == vindta_path + "2018/2018_Aug_RWS_CO2.dbs"),
    "k_dic_good",
] = False
dbs.loc[
    (dbs.k_dic_here > 0.0114)
    & (dbs.dbs_fname == vindta_path + "2019/2019_02_RWS_CO2.dbs"),
    "k_dic_good",
] = False
dbs.loc[
    (dbs.k_dic_here < 0.01104)
    & (dbs.dbs_fname == vindta_path + "2019/2019_11_RWS_CO2.dbs"),
    "k_dic_good",
] = False
dbs.loc[
    (dbs.dic_cell_id == "C_May03-19_0705") & (dbs.bottle == "CRM#171_1037"),
    "k_dic_good",
] = False
# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~

# Calibrate DIC!
dbs.calibrate_dic()

# ~~~ QC DIC measurements: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dbs["dic_good"] = ~np.isnan(dbs.dic)
dbs.loc[
    (dbs.bottle == "ROTTMPT70_2019006224") & (dbs.counts < 15000), "dic_good"
] = False
dbs.loc[dbs.bottle == "WALCRN2_2018005578", "dic_good"] = False
dbs.loc[dbs.bottle == "TERSLG235_2018005745", "dic_good"] = False
# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~


def dbs2data(dbs, data, fields):
    """Transfer results from dbs into data, taking means for duplicates."""
    if isinstance(fields, str):
        fields = [fields]
    ufields = ["dic_cell_id"]
    # numfields = ["dic", "salinity_v6", "alkalinity", "emf0", "pH_vindta_free_lab", "pH_vindta_temperature"]
    for field in fields:
        data[field] = np.nan
        if field not in ufields:
            data[field + "_std"] = np.nan
        data[field + "_count"] = 0
        if (field + "_good") not in dbs.columns:
            dbs[field + "_good"] = True
    for row in data.index:
        dr = data.loc[row]
        # if dr.station_bottleid in dbs.bottle.values:
        if dr.name in dbs.data_index.values:
            for field in fields:
                dbs_iloc = np.where(
                    # (dr.station_bottleid == dbs.bottle.values) & dbs[field + "_good"]
                    (dr.name == dbs.data_index.values)
                    & dbs[field + "_good"]
                )[0]
                if np.size(dbs_iloc) > 0:
                    fdata = dbs.iloc[dbs_iloc][field]
                    if field in ufields:
                        data.loc[row, field] = np.unique(fdata)[0]
                    else:
                        data.loc[row, field] = np.mean(fdata)
                        if np.size(dbs_iloc) > 1:
                            data.loc[row, field + "_std"] = np.std(fdata)
                    data.loc[row, field + "_count"] = np.size(dbs_iloc)
    return data


# Put good DIC results back into data table, averaging duplicates
data = dbs2data(dbs, data, "dic")

# Plot CRM calibration factors through time
if draw_figures:
    dbs.plot_k_dic(figure_path="figures/calibration")
    dbs.plot_dic_offset(figure_path="figures/calibration")
    for session in dbs.sessions.index:
        sub = dbs.subset(dbs.dic_cell_id == session)
        if any(~np.isnan(sub.dic_offset)):
            sub.plot_dic_offset(
                figure_path="figures/calibration/dic_offset",
                figure_format="{}.png".format(session),
            )
            plt.close()

# Apply mercuric chloride correction to DIC in data (but not in dbs)
data["dic_poisoned"] = copy.deepcopy(data.dic)
data["dic"] = ks.poison_correction(data.dic_poisoned, sample_volume, poison_volume)

#%% Calkulate total alkalinity =========================================================
#   ___           _    _             _           _              _____  _____
#  (  _`\        (_ ) ( )           (_ )        ( )_           (_   _)(  _  )
#  | ( (_)   _ _  | | | |/')  _   _  | |    _ _ | ,_)   __       | |  | (_) |
#  | |  _  /'_` ) | | | , <  ( ) ( ) | |  /'_` )| |   /'__`\     | |  |  _  |
#  | (_( )( (_| | | | | |\`\ | (_) | | | ( (_| || |_ (  ___/     | |  | | | |
#  (____/'`\__,_)(___)(_) (_)`\___/'(___)`\__,_)`\__)`\____)     (_)  (_) (_)

# Set up titration table (tdata) for Calkulate
dbs2calkulate = {  # keys = dbs column names, values = titration table column names
    "bottle": "bottle",
    "salinity": "salinity",
    "alkalinity_certified": "alkalinity_certified",
    "dbs_fname": "analysis_batch",
    "dic": "dic",
    "ammonia": "total_ammonia",
    "phosphate": "total_phosphate",
    "silicate": "total_silicate",
    "sulfate": "total_sulfate",
    "analysis_datetime": "analysis_datetime",
}
tdata = dbs[dbs2calkulate].rename(columns=dbs2calkulate)
tdata["analyte_volume"] = 95.0  # check this with Sharyn! <--------------------- TO DO!
tdata["file_path"] = tdata.analysis_batch.apply(lambda x: x.replace(".dbs", "/"))
tdata["file_name"] = dbs.apply(
    lambda x: "{}-{}  {}  ({}){}.dat".format(
        x.station, x.cast, x.niskin, x.depth, x.bottle
    ),
    axis=1,
)

# Override temperatures if requested
if temperature_override is not None:
    tdata["temperature_override"] = temperature_override

# Use settings that are consistent with later PyCO2SYS calculations
tdata["opt_k_bisulfate"] = opt_k_bisulfate
tdata["opt_total_borate"] = opt_total_borate
tdata["opt_k_carbonic"] = opt_k_carbonic
tdata["opt_k_fluoride"] = opt_k_fluoride

# ~~~ QC the titration files: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Files to ignore
tdata["file_good"] = True
tdata.loc[
    np.isin(
        tdata.apply(lambda x: x.file_path + x.file_name, axis=1),
        [
            "data/VINDTA#15 Data/2019/2019_08_RWS_CO2/0-0  0  (0)TEST1.dat",
            "data/VINDTA#15 Data/2019/2019_11_RWS_CO2/0-0  0  (0)JUNK6.dat",
            "data/VINDTA#15 Data/2019/2019_02_RWS_CO2/0-0  0  (0)JUNK7.dat",
            "data/VINDTA#15 Data/2020/2020_07_RWS_CO2/0-0  0  (0)JUNK7.dat",
        ],
    ),
    "file_good",
] = False

# Files with different names
tdata.loc[
    tdata.file_name == "11-6  18  (0)WALCRN20_2018005589.dat", "file_name"
] = "11-6  18  (0)WALCRN20_2018005589B.dat"
tdata.loc[
    tdata.file_name == "27-5  19  (0)NOORDWK20_2019006697.dat", "file_name"
] = "27-5  19  (0)NOORDWK20_2019006697B.dat"
tdata.loc[
    tdata.file_name == "5-0  0  (0)NUTSL .dat", "file_name"
] = "5-0  0  (0)NUTSL.dat"
tdata.loc[
    tdata.file_path + tdata.file_name
    == "data/VINDTA#15 Data/2019/2019_02_RWS_CO2/12-12  18  (0)TERSLG_2018005727.dat",
    "file_name",
] = "12-12  18  (0)TERSLG100C_2018005727.dat"
tdata.loc[
    tdata.file_path + tdata.file_name
    == "data/VINDTA#15 Data/2019/2019_08_RWS_CO2/666-0  0  (0)CRM171-0057A.dat",
    "file_name",
] = "666-0  0  (0)CRM0057B.dat"
tdata.loc[
    tdata.file_name == "13-5  20  (0)TERSLG135_2020005221.dat", "file_name"
] = "13-5  20  (0)TERSLG135_2020005221A.dat"

# CRMs not to use for calibration
tdata["reference_good"] = ~np.isnan(tdata.alkalinity_certified)
tdata.loc[
    np.isin(
        dbs.bottle,
        [
            "CRM#171-0745",
            "CRM#171-0745B",
            "CRM171_0207A",
        ],
    ),
    "reference_good",
] = False

# Mark bad dat files to silence all Calkulate warnings
tdata["file_good"] &= ~np.isnan(tdata.salinity)
tdata.loc[
    np.isin(
        tdata.file_name,
        [
            "666-0  0  (0)CRM171-0057A.dat",
            "666-0  0  (0)CRM171_0207B.dat",
            "12-12  18  (0)TERSLG_2018005727.dat",
            "666-0  0  (0)CRM#171-0745.dat",
            "666-0  0  (0)CRM#171-0745B.dat",
        ],
    ),
    "file_good",
] = False
# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~

# Convert tdata to Calkulate titration dataset, calibrate and solve and return to dbs
tdata = calk.dataset.calkulate(tdata)  # for calk23
tdata["alkalinity_offset"] = tdata.alkalinity - tdata.alkalinity_certified
dbs["alkalinity"] = tdata.alkalinity
dbs["emf0"] = tdata.emf0
dbs["pH_vindta_free_lab"] = tdata.pH_initial
dbs["pH_vindta_temperature"] = tdata.temperature_initial  # for calk23
dbs["titrant_molinity"] = tdata.titrant_molinity
dbs["titrant_molinity_here"] = tdata.titrant_molinity_here

# ~~~ QC alkalinity measurements: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dbs["alkalinity_good"] = ~np.isnan(dbs.alkalinity)
dbs.loc[
    (dbs.bottle == "NOORDWK20") & (dbs["dic_cell_id"] == "C_Aug12-19_0708"),
    "alkalinity_good",
] = False
dbs["emf0_good"] = dbs.alkalinity_good
dbs["pH_vindta_free_lab_good"] = dbs.alkalinity_good
dbs["pH_vindta_temperature_good"] = dbs.alkalinity_good
dbs.loc[
    (dbs.bottle == "WALCRN2_2018005578") & (dbs.pH_vindta_free_lab < 7),
    "pH_vindta_free_lab_good",
] = False
# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~

# Put good alkalinity results back into data table, averaging duplicates
data = dbs2data(
    dbs,
    data,
    [
        "alkalinity",
        "emf0",
        "pH_vindta_free_lab",
        "pH_vindta_temperature",
        "analysis_datetime",
    ],
)
data["analysis_datetime"] = pd.to_datetime(data.analysis_datetime)

# Apply mercuric chloride correction to alkalinity in data (but not in dbs or tdata)
data["alkalinity_poisoned"] = copy.deepcopy(data.alkalinity)
data["alkalinity"] = ks.poison_correction(
    data.alkalinity_poisoned, sample_volume, poison_volume
)

# Plot alkalinity calibration
if draw_figures:
    fig, axs = plt.subplots(nrows=2, dpi=300, figsize=(6, 8))
    ax = axs[0]
    tdata[tdata.reference_good].plot.scatter(
        "analysis_datetime",
        "alkalinity_offset",
        ax=ax,
        c="xkcd:navy",
        alpha=0.6,
        label="Used",
    )
    tdata[~tdata.reference_good].plot.scatter(
        "analysis_datetime",
        "alkalinity_offset",
        ax=ax,
        c="xkcd:strawberry",
        alpha=0.8,
        label="Ignored",
    )
    ax.axhline(0, c="k", linewidth=0.8)
    ax.grid(alpha=0.3)
    ax.set_xlabel("")
    ax.set_ylabel("CRM alkalinity (meas. − cert.) / μmol$\cdot$kg$^{-1}$")
    ax.legend()
    ax = axs[1]
    tdata[tdata.reference_good].plot.scatter(
        "analysis_datetime",
        "titrant_molinity_here",
        ax=ax,
        c="xkcd:navy",
        alpha=0.6,
        label="Used",
    )
    tdata[~tdata.reference_good].plot.scatter(
        "analysis_datetime",
        "titrant_molinity_here",
        ax=ax,
        c="xkcd:strawberry",
        alpha=0.8,
        label="Ignored",
    )
    ax.grid(alpha=0.3)
    ax.set_xlabel("")
    ax.set_ylabel("Acid molinity / mol$\cdot$kg$^{-1}$")
    ax.legend()
    plt.tight_layout()
    if save_figures:
        plt.savefig("figures/calibration/alkalinity_crms.png")

#%% Determine spectrophotometric pH ====================================================
#   ___                         _                          _   _
#  (  _`\                      ( )_                       ( ) ( )
#  | (_(_) _ _      __     ___ | ,_) _ __   _       _ _   | |_| |
#  `\__ \ ( '_`\  /'__`\ /'___)| |  ( '__)/'_`\    ( '_`\ |  _  |
#  ( )_) || (_) )(  ___/( (___ | |_ | |  ( (_) )   | (_) )| | | |
#  `\____)| ,__/'`\____)`\____)`\__)(_)  `\___/'   | ,__/'(_) (_)
#         | |                                      | |
#         (_)                                      (_)

# Define pH file names (for Excel files prepared by Sharyn/Karel)
pH_root = "data/pH_spectro/pH Data "
pH_files = [
    pH_root + pH_file
    for pH_file in [
        "2018/2018_JUNE_RWS_PH/2018_JUNE_RWS_PH_JAN_FEB_MAR_APR.xlsx",
        "2018/2018_AUG_RWS_PH/2018_AUG_RWS_PH_MAY_JUN_JUL_MPH.xlsx",
        "2018/2018_Nov_RWS_pH/2018_NOV_RWS_PH_SEP_OCT.xlsx",
        "2019/2019_02_RWS_pH/2019_02_RWS_pH_Results_MPH.xlsx",
        "2019/2019_08_RWS_Ph_with Correct Salinities/2019_08_RWS_Ph_Results_MPH.xlsx",
        "2019/2019_11_RWS_pH/2019_11_RWS_pH_MPH.xlsx",
        "2020/2020_07_RWS_pH/2020_07_RWS_PH_MPH.xlsx",
        # DO NOT ADD MORE FILES HERE --- go to section starting `new_pHdata = []` below!
    ]
]

# Prepare to import spectrophotometric pH data
rename_mapper = {
    "Sample Name": "station_bottleid",
    "Temp (25)": "temperature_analysis",
    "Weight(25)": "temperature_analysis",
    "Temperature": "temperature_analysis",
    "Salinity (35)": "salinity_lab",
    "Volume(35)": "salinity_lab",
    "Salinity": "salinity_lab",
    "pH": "pH_raw",
    "Abs<578nm>": "absorbance_578nm",
    "Abs<434nm>": "absorbance_434nm",
    "Abs<730nm>": "absorbance_730nm",
}
keep_cols = list(set(rename_mapper.values())) + ["fname", "type", "date_analysis"]

# ~~~ Import spectrophotometric pH data from lab: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import spectro pH data #0: June 2018
pdata0 = pd.read_excel(pH_files[0], skiprows=list(range(17)) + [18], nrows=272)
pdata0.rename(mapper=rename_mapper, axis=1, inplace=True)
pdata0["fname"] = pH_files[0]
pdata0["type"] = "?"
pdata0["date_analysis"] = "2018/06/29"

# Import spectro pH data #1: August 2018
pdata1 = pd.read_excel(pH_files[1], skiprows=list(range(16)) + [17], nrows=231)
pdata1.rename(mapper=rename_mapper, axis=1, inplace=True)
pdata1["fname"] = pH_files[1]
pdata1["type"] = "?"
pdata1["date_analysis"] = "2018/08/16"

# Import spectro pH data #2: November 2018
pdata2 = pd.read_excel(
    pH_files[2], skiprows=list(range(16)) + [17], nrows=173, sheet_name="Work Sheet"
)
pdata2.rename(mapper=rename_mapper, axis=1, inplace=True)
namespl = re.compile("\d{1,3}\s{1,2}")
pdata2["sample_name"] = pdata2.apply(
    lambda x: namespl.split(x["#  Name             "])[1], axis=1
)
pdata2["station"] = pdata2.apply(
    lambda x: x.sample_name.split("_")[0].split("-")[0], axis=1
)
pdata2["Sample ID"].fillna(method="pad", inplace=True)
pdata2["station_bottleid"] = pdata2.apply(
    lambda x: x.station + "_" + str(x["Sample ID"]), axis=1
)
pdata2["fname"] = pH_files[2]
pdata2["type"] = "?"
pdata2["date_analysis"] = "2018/11/27"

# Import spectro pH data #3: February 2019
pdata3 = pd.read_excel(
    pH_files[3], skiprows=list(range(16)) + [17], nrows=265, sheet_name="Work Sheet1"
)
pdata3.rename(mapper={"Salinity": "junk"}, axis=1, inplace=True)
pdata3.rename(mapper=rename_mapper, axis=1, inplace=True)
pdata3["ID"] = pdata3.ID.fillna(method="pad")
pdata3["Place"] = pdata3.Place.fillna(method="pad")
pdata3["Place"] = pdata3.Place.apply(
    lambda x: x.split(" - D")[0].split(" -D")[0].split("-D")[0]
)
pdata3["station_bottleid"] = pdata3.apply(lambda x: x.Place + "_" + str(x.ID), axis=1)
pdata3["fname"] = pH_files[3]
pdata3["type"] = "?"
pdata3["date_analysis"] = np.where(pdata3["#"] <= 168, "2019/05/07", "2019/05/08")

# Import spectro pH data #4: August 2019
pdata4 = pd.read_excel(
    pH_files[4], skiprows=list(range(17)) + [18], nrows=567, sheet_name="Work Sheet1"
)
pdata4.rename(mapper=rename_mapper, axis=1, inplace=True)
pdata4["ID"] = pdata4.ID.fillna(method="pad")
pdata4["Place"] = pdata4.Place.fillna(method="pad")
pdata4["Place"] = pdata4.Place.apply(lambda x: str(x).split(" R")[0])
pdata4["station_bottleid"] = pdata4.apply(
    lambda x: str(x.Place) + "_" + str(x.ID), axis=1
)
pdata4["fname"] = pH_files[4]
pdata4["type"] = "?"
pdata4["date_analysis"] = "2019/08/22"

# Import spectro pH data #5: November 2019
pdata5 = pd.read_excel(
    pH_files[5], skiprows=list(range(16)) + [17], nrows=217, sheet_name="Work Sheet1"
)
pdata5.rename(mapper=rename_mapper, axis=1, inplace=True)
pdata5_key = pd.read_excel(
    pH_files[5], skiprows=[1, 2], nrows=53, sheet_name="Results pH 2019_11"
)
pdata5_key["station_bottleid"] = pdata5_key.apply(
    lambda x: str(x.Place) + "_" + str(x.ID), axis=1
)
pdata5_key["analysis_name"] = pdata5_key["Analysis Name"].apply(lambda x: str(x)[:-1])
pdata5["station_bottleid"] = pdata5.Name.apply(
    lambda x: pdata5_key.station_bottleid[pdata5_key.analysis_name == x[:-1]].values[0]
)
pdata5["fname"] = pH_files[5]
pdata5["type"] = "?"
pdata5["date_analysis"] = "2019/11/28"

# Import spectro pH data #6: July 2020
pdata6 = pd.read_excel(pH_files[6], skiprows=18, sheet_name="Work Sheet1", nrows=294)
pdata6["fname"] = pH_files[6]
pdata6["type"] = "?"
pdata6["date_analysis"] = "2020/07/21"

# Import subsequent batches of pH data with consistent spreadsheet formatting
new_pHdata_path = "data/pH_spectro/"
new_pHdata_files = {
    "2020_10_RWS_PH_MPH.xlsx": "2020/10/28",
    "2021_05_RWS_pH_MPH.xlsx": "2021/05/06",
    "2021_06_RWS_pH_Final Data_MPH Format_MPH.xlsx": "2021/06/29",
}
new_pHdata = []
for file, date_analysis in new_pHdata_files.items():
    pHdata = pd.read_excel(new_pHdata_path + file, skiprows=18)
    pHdata = pHdata[pHdata.flag != 1]
    pHdata["fname"] = new_pHdata_path + file
    pHdata["date_analysis"] = date_analysis
    new_pHdata.append(pHdata)

# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~

# Join all files together and recalculate pH
pdata = pd.concat(
    [
        pdata[keep_cols]
        for pdata in [
            pdata0,
            pdata1,
            pdata2,
            pdata3,
            pdata4,
            pdata5,
            pdata6,
            *new_pHdata,
        ]
    ]
)
pdata.reset_index(drop=True, inplace=True)
pdata.date_analysis = pd.to_datetime(pdata.date_analysis, format="%Y/%m/%d")


def get_spectro_salinity(x):
    L = data.station_bottleid == x.station_bottleid
    if any(L):
        return data.salinity[L].values[0]
    elif str(x.station_bottleid).startswith("CRM#171"):
        return ks.crm.dickson_certified_values[171].salinity
    elif str(x.station_bottleid).startswith("TRIS"):
        return 35
    else:
        return x.salinity_lab


pdata["salinity"] = pdata.apply(get_spectro_salinity, axis=1)
pdata["pH_spectro_total_lab"] = ks.spectro.pH_NIOZ(
    pdata.absorbance_578nm,
    pdata.absorbance_434nm,
    pdata.absorbance_730nm,
    temperature=pdata.temperature_analysis,
    salinity=pdata.salinity,
)
pdata["station_bottleid"] = pdata.station_bottleid.apply(
    lambda x: str(x).replace("-", "_")
)
pdata["is_sample"] = pdata.station_bottleid.apply(
    lambda x: str(x).split("_")[0] in stations.index
)

# ~~~ QC lab pH measurements: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pdata["pH_good"] = True
pdata.loc[
    (pdata.station_bottleid == "TERSLG10_2019007106") & (pdata.salinity_lab == 1),
    "pH_good",
] = False
pdata.loc[
    pdata.station_bottleid == "ROTTMPT50_2018005770", "station_bottleid"
] = "ROTTMPT50_2018008770"
pdata.loc[
    pdata.station_bottleid == "ROTTMPT50_201800771", "station_bottleid"
] = "ROTTMPT50_2018008771"
pdata.loc[
    pdata.station_bottleid == "ROTTMPT70_2018005777", "station_bottleid"
] = "ROTTMPT70_2018008777"
# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~

# Aggregate good pH measurements
pH_spectro_total_lab = (
    pdata[pdata.is_sample & pdata.pH_good]
    .groupby("station_bottleid")
    .pH_spectro_total_lab.agg([np.mean, np.median, np.std, np.size])
    .rename(mapper={"mean": "pH_mean", "median": "pH_median", "std": "pH_std"}, axis=1)
)
pH_spectro_total_lab["date_analysis"] = (
    pdata.loc[pdata.is_sample & pdata.pH_good]
    .groupby("station_bottleid")
    .apply(lambda x: np.mean(x.date_analysis))
)
pH_spectro_temperature = (
    pdata[pdata.is_sample & pdata.pH_good]
    .groupby("station_bottleid")
    .temperature_analysis.mean()
)


def get_pH_spectro(x):
    if x.station_bottleid in pH_spectro_total_lab.index:
        x_mean = pH_spectro_total_lab.loc[x.station_bottleid].pH_mean
        x_std = pH_spectro_total_lab.loc[x.station_bottleid].pH_std
        x_count = pH_spectro_total_lab.loc[x.station_bottleid].size
        x_temperature = pH_spectro_temperature.loc[x.station_bottleid]
    else:
        x_mean = np.nan
        x_std = np.nan
        x_count = np.nan
        x_temperature = np.nan
    return pd.Series(
        {
            "pH_spectro_total_lab": x_mean,
            "pH_spectro_total_lab_std": x_std,
            "pH_spectro_total_lab_count": x_count,
            "pH_spectro_temperature": x_temperature,
        }
    )


def get_pH_spectro_date(x):
    if x.station_bottleid in pH_spectro_total_lab.index:
        x_date = pH_spectro_total_lab.loc[x.station_bottleid].date_analysis
    else:
        x_date = np.datetime64("NaT")
    return pd.Series(
        {
            "pH_spectro_date_analysis": x_date,
        }
    )


# Put results into data
pH_to_data = data.apply(get_pH_spectro, axis=1)
for col, col_data in pH_to_data.iteritems():
    data[col] = col_data
data["pH_spectro_date_analysis"] = data.apply(get_pH_spectro_date, axis=1)

# Check for not-assigned pH results
pH_not_assigned = [
    sbid
    for sbid in pH_spectro_total_lab.index
    if sbid not in data.station_bottleid.values
]

#%% Process Caribbean monitoring data ==================================================
#   ___                    _      _
#  (  _`\               _ ( )    ( )
#  | ( (_)   _ _  _ __ (_)| |_   | |_      __     _ _   ___
#  | |  _  /'_` )( '__)| || '_`\ | '_`\  /'__`\ /'_` )/' _ `\
#  | (_( )( (_| || |   | || |_) )| |_) )(  ___/( (_| || ( ) |
#  (____/'`\__,_)(_)   (_)(_,__/'(_,__/'`\____)`\__,_)(_) (_)

# # Import data from 64PE465 cruise
# pe465 = pd.read_csv("data/PE465-rws.csv")
# pe465.rename(
#     columns={
#         "phos": "total_phosphate",
#         "si": "total_silicate",
#         "nh4": "total_ammonia",
#         "ta": "alkalinity",
#     },
#     inplace=True,
# )
# pe465["longitude_w"] = -pe465.longitude
# pe465["datetime"] = pd.to_datetime(
#     mdates.num2date(pe465.ndateUTC + mdates.date2num(np.datetime64("0000-01-01")) - 1)
# )
# pe465["ndateUTC"] = mdates.date2num(pe465.datetime)
# pe465["date"] = pd.to_datetime(mdates.num2date(np.floor(pe465.ndateUTC)))

# # Begin setting up carib dicts to convert to xarray Dataset with 64PE465 data
# pe465vars = [
#     "dic",
#     "alkalinity",
#     "total_phosphate",
#     "total_silicate",
#     "total_ammonia",
#     "salinity",
#     "temperature",
# ]
# carib = {
#     var: [["station", "datetime"], np.vstack(pe465[var].to_numpy())]
#     for var in pe465vars
# }
# carib.update(
#     {var: ("station", pe465[var].to_numpy()) for var in ["latitude", "longitude"]}
# )
# carib_coords = {
#     "station": ("station", pe465.station_rws.to_numpy()),
#     "datetime": ["datetime", np.array([mdates.date2num(pe465.date.to_numpy()[0])])],
# }

# # Add newer RWS monitoring data
# pdata_carib = (
#     pdata.loc[pdata.type == "SS", ["station_bottleid", "pH_spectro_total_lab"]]
#     .groupby("station_bottleid")
#     .mean()
# )
# date_raw = [d.split("_")[1] for d in pdata_carib.index]
# pdata_carib["date"] = pd.to_datetime(
#     ["20{}{}-{}{}-{}{}".format(d[4], d[5], d[2], d[3], d[0], d[1]) for d in date_raw]
# )
# pdata_carib["ndate"] = mdates.date2num(pdata_carib.date)
# pdata_carib["station"] = [
#     int(s[0].split("RWS")[1]) for s in pdata_carib.index.str.split("-")
# ]
# for var in pe465vars:
#     carib[var][1] = np.append(
#         carib[var][1], np.full((np.size(carib_coords["station"][1]), 2), np.nan), axis=1
#     )
# carib_coords["datetime"][1] = np.append(
#     carib_coords["datetime"][1], mdates.date2num(pdata_carib.date.unique())
# )
# carib["pH_spectro_total_lab"] = [
#     ["station", "datetime"],
#     np.full_like(carib["dic"][1], np.nan),
# ]
# for i, row in pdata_carib.iterrows():
#     carib["pH_spectro_total_lab"][1][
#         carib_coords["station"][1] == row.station,
#         carib_coords["datetime"][1] == row.ndate,
#     ] = row.pH_spectro_total_lab

# # Convert to xarray Dataset
# carib = {k: tuple(v) for k, v in carib.items()}
# carib_coords["datetime"] = ("datetime", mdates.num2date(carib_coords["datetime"][1]))
# xarib = xr.Dataset(carib, coords=carib_coords)

# # Set attributes
# xarib.pH_spectro_total_lab.attrs.update({"long_name": "pH$_\mathrm{T}$ @ 25 °C"})
# xarib.temperature.attrs.update({"long_name": "Temperature / °C"})
# xarib.salinity.attrs.update({"long_name": "Practical salinity"})

# # Calculate with PyCO2SYS (temporary)
# results = pyco2.sys(
#     xarib.dic.data, xarib.alkalinity.data, 2, 1, salinity=xarib.salinity.data
# )
# xarib["pH25"] = (
#     ["station", "datetime"],
#     np.where(
#         np.isnan(xarib.pH_spectro_total_lab.data),
#         results["pH"],
#         xarib.pH_spectro_total_lab.data,
#     ),
# )
# xarib.pH25.attrs.update({"long_name": "pH$_\mathrm{T}$ @ 25 °C"})

# # Plot pH
# fig, ax = plt.subplots(dpi=300)
# xarib.pH25.plot(ax=ax, cmap="viridis_r")
# ax.set_ylabel("Station number")
# ax.set_xlabel("")
# locator = mdates.AutoDateLocator(minticks=3, maxticks=9)
# formatter = mdates.ConciseDateFormatter(locator)
# ax.xaxis.set_major_locator(locator)
# ax.xaxis.set_major_formatter(formatter)
# ax.xaxis.get_offset_text().set_visible(False)
# # plt.tight_layout()
# plt.savefig("figures/pH25-carib.png")

#%% Consistency checks and adjustments =================================================
#   ___                                 _
#  (  _`\                      _       ( )_
#  | ( (_)   _     ___    ___ (_)  ___ | ,_)   __    ___     ___  _   _
#  | |  _  /'_`\ /' _ `\/',__)| |/',__)| |   /'__`\/' _ `\ /'___)( ) ( )
#  | (_( )( (_) )| ( ) |\__, \| |\__, \| |_ (  ___/| ( ) |( (___ | (_) |
#  (____/'`\___/'(_) (_)(____/(_)(____/`\__)`\____)(_) (_)`\____)`\__, |
#                                                                ( )_| |
#                                                                `\___/'

# Import finalised data from previous version
prev = pd.read_excel("results/RWS-NIOZ North Sea data v6-1 for SDG14-3-1.xlsx")
dbs_v6 = pd.read_csv("pickles/dbs_v6.csv")
for var in ["salinity", "titrant_molinity", "titrant_molinity_here"]:
    dbs[var + "_v6"] = dbs_v6[var]
    dbs[var + "_v6_good"] = ~np.isnan(dbs[var + "_v6"])
    data = dbs2data(dbs, data, var + "_v6")
    if var != "salinity":
        data = dbs2data(dbs, data, var)
data = dbs2data(dbs, data, "dic_cell_id")
rounding = {
    "dic": 1,
    "nitrate": 2,
    "nitrite": 2,
    "salinity": 2,
    "temperature": 2,
    "ammonia": 2,
    "phosphate": 2,
    "silicate": 2,
    "doc": 1,
    "alkalinity": 1,
    "pH_spectro_total_lab": 3,
    "pH_spectro_temperature": 1,
}  # how many decimal places to round to for each consistency check

# Copy previously released data into new data DataFrame
for var in rounding:
    var_prev = "{}_previous".format(var)
    data[var_prev] = np.nan
    for i, row in data.iterrows():
        l = (row.bottleid == prev.bottle_id) & (row.station == prev.station)
        if any(l):
            assert sum(l) == 1
            if prev[l][var].values != -999:
                data.loc[i, var_prev] = prev[l][var].values
data["zeroes"] = 0


def get_differences(data):
    for var in rounding:
        var_prev = "{}_previous".format(var)
        data["{}_previous_diff".format(var)] = (
            np.round(data[var], rounding[var]) - data[var_prev]
        )
        data["{}_changed".format(var)] = (
            data["{}_previous_diff".format(var)] != 0
        ) & ~np.isnan(data["{}_previous_diff".format(var)])
        data["{}_lost".format(var)] = np.isnan(data[var]) & ~np.isnan(data[var_prev])
        data["{}_gained".format(var)] = ~np.isnan(data[var]) & np.isnan(data[var_prev])
    return data


data = get_differences(data)

# Assign reasons for changes
data["dic_changed_salinity"] = data.dic_changed & (
    data.salinity_changed
    | (data.salinity != np.round(data.salinity_v6, rounding["salinity"]))
)
data["dic_changed_recalibration"] = data.dic_changed & (
    data.dic_cell_id == "C_May03-19_0705"
)
data["dic_changed_replicate"] = data.dic_changed & data.dic_count > 1
data["alkalinity_changed_dic"] = data.alkalinity_changed & data.dic_changed
data["pH_spectro_total_lab_changed_salinity"] = (
    data.pH_spectro_total_lab_changed & data.salinity_changed
)

# Update alkalinity values to match original v6.1 publication for negligible changes
# due to solver tolerances
if "alkalinity_new" not in data:
    data["alkalinity_new"] = copy.deepcopy(data.alkalinity)
    data["alkalinity"] = np.where(
        data.alkalinity_changed & ~data.alkalinity_changed_dic,
        data.alkalinity_previous,
        data.alkalinity_new,
    )
data = get_differences(data)

# Plot the (hopefully absent...) differences
varlabels = {
    "alkalinity": "alkalinity / μmol$\cdot$kg$^{-1}$",
    "dic": "DIC / μmol$\cdot$kg$^{-1}$",
    "pH_spectro_total_lab": "pH$_\mathrm{T}$ @ lab temperature",
    "pH_spectro_temperature": "pH$_\mathrm{T}$ spectro. temperature",
}
if draw_figures:
    for var in rounding:
        fxvar = "datetime"
        fyvar = var
        fig, ax = plt.subplots(dpi=300)
        l = ~data["{}_changed".format(fyvar)]
        ax.scatter(
            fxvar,
            "{}_previous_diff".format(fyvar),
            data=data[l],
            s=10,
            c="xkcd:navy",
            alpha=0.5,
            label="Unchanged ({})".format(l.sum()),
        )
        if var == "dic":
            l = data["{}_changed_salinity".format(fyvar)]
            ax.scatter(
                fxvar,
                "{}_previous_diff".format(fyvar),
                data=data[l],
                s=15,
                c="xkcd:grass",
                alpha=0.5,
                marker="^",
                label="Salinity ({})".format(l.sum()),
            )
            l = data["{}_changed_recalibration".format(fyvar)]
            ax.scatter(
                fxvar,
                "{}_previous_diff".format(fyvar),
                data=data[l],
                s=15,
                c="xkcd:royal purple",
                alpha=0.5,
                marker="v",
                label="Recalibrated ({})".format(l.sum()),
            )
            l = data["{}_changed_replicate".format(fyvar)]
            ax.scatter(
                fxvar,
                "{}_previous_diff".format(fyvar),
                data=data[l],
                s=15,
                c="xkcd:dark yellow",
                alpha=0.5,
                marker="x",
                label="Replicate ({})".format(l.sum()),
            )
            l = (
                data["{}_changed".format(fyvar)]
                & ~data["{}_changed_salinity".format(fyvar)]
                & ~data["{}_changed_recalibration".format(fyvar)]
                & ~data["{}_changed_replicate".format(fyvar)]
            )
            ax.scatter(
                fxvar,
                "{}_previous_diff".format(fyvar),
                data=data[l],
                s=15,
                c="xkcd:strawberry",
                alpha=0.5,
                label="Changed ({})".format(l.sum()),
                marker="s",
            )
        elif var == "alkalinity":
            l = data["{}_changed_dic".format(fyvar)]
            ax.scatter(
                fxvar,
                "{}_previous_diff".format(fyvar),
                data=data[l],
                s=15,
                c="xkcd:grass",
                alpha=0.5,
                marker="^",
                label="DIC/salinity ({})".format(l.sum()),
            )
            l = data["{}_changed".format(fyvar)] & ~data["{}_changed_dic".format(fyvar)]
            ax.scatter(
                fxvar,
                "{}_previous_diff".format(fyvar),
                data=data[l],
                s=15,
                c="xkcd:strawberry",
                alpha=0.5,
                label="Changed ({})".format(l.sum()),
                marker="s",
            )
        elif var == "pH_spectro_total_lab":
            l = data["{}_changed_salinity".format(fyvar)]
            ax.scatter(
                fxvar,
                "{}_previous_diff".format(fyvar),
                data=data[l],
                s=15,
                c="xkcd:grass",
                alpha=0.5,
                marker="^",
                label="Salinity ({})".format(l.sum()),
            )
            l = (
                data["{}_changed".format(fyvar)]
                & ~data["{}_changed_salinity".format(fyvar)]
            )
            ax.scatter(
                fxvar,
                "{}_previous_diff".format(fyvar),
                data=data[l],
                s=15,
                c="xkcd:strawberry",
                alpha=0.5,
                label="Changed ({})".format(l.sum()),
                marker="s",
            )
        else:
            l = data["{}_changed".format(fyvar)]
            ax.scatter(
                fxvar,
                "{}_previous_diff".format(fyvar),
                data=data[l],
                s=10,
                c="xkcd:strawberry",
                alpha=0.5,
                label="Changed ({})".format(l.sum()),
            )
        l = data["{}_lost".format(fyvar)]
        ax.scatter(
            fxvar,
            "zeroes",
            data=data[l],
            label="Lost ({})".format(l.sum()),
            s=40,
            c="xkcd:orange",
            alpha=0.8,
            marker="8",
            zorder=0,
        )
        # yl = ax.get_ylim()
        # data["yl"] = yl[0]
        l = data["{}_gained".format(fyvar)]
        ax.scatter(
            fxvar,
            # "yl",
            "zeroes",
            data=data[l],
            label="Gained ({})".format(l.sum()),
            s=40,
            c="xkcd:sky blue",
            alpha=0.8,
            marker="d",
            zorder=-1,
        )
        # ax.set_ylim(yl)
        ax.axhline(0, c="k", linewidth=0.8)
        ax.legend(edgecolor="k")
        ax.grid(alpha=0.3)
        ax.set_xlabel("Sampling date")
        if fyvar in varlabels:
            varlabel = varlabels[fyvar]
        else:
            varlabel = fyvar
        ax.set_ylabel("{} (v10 $−$ v6.1)".format(varlabel))
        ax.set_title("Changes in v10 since v6.1 release")
        # ax.set_ylim([-1, 1])
        plt.tight_layout()
        plt.savefig("figures/calibration/comparison/{}.png".format(fyvar))


#%% Import QuAAtro data from Karel =====================================================
#   _____         _____  _____  _
#  (  _  )       (  _  )(  _  )( )_
#  | ( ) | _   _ | (_) || (_) || ,_) _ __   _
#  | | | |( ) ( )|  _  ||  _  || |  ( '__)/'_`\
#  | (('\|| (_) || | | || | | || |_ | |  ( (_) )
#  (___\_)`\___/'(_) (_)(_) (_)`\__)(_)  `\___/'

# Import Karel's worksheet
quaatro = pd.read_excel(
    "data/QuAAtro/201103 RWS COMP DIC TAlk AR1 DAMP 8SEC MPH.xlsx",
    sheet_name="RESULTS",
    skiprows=11,
)[
    [
        "name_karel",
        "salinity_karel",
        "dic_vol",
        "dic_certified",
        "alkalinity_bphb_vol",
        "alkalinity_bphb_certified",
        "alkalinity_methyl_vol",
        "alkalinity_methyl_certified",
    ]
]

# Get metadata
quaatro["temperature_analysis"] = 23.0
quaatro["density_analysis"] = calk.density.seawater_1atm_MP81(
    temperature=quaatro.temperature_analysis, salinity=quaatro.salinity_karel
)

# Convert to substance content and calibrate
qvars = ["dic", "alkalinity_bphb", "alkalinity_methyl"]
for v in qvars:
    quaatro["{}_raw".format(v)] = quaatro["{}_vol".format(v)] / quaatro.density_analysis
    quaatro["{}_offset_raw".format(v)] = (
        quaatro["{}_raw".format(v)] - quaatro["{}_certified".format(v)]
    )
    quaatro[v] = quaatro["{}_raw".format(v)] - quaatro["{}_offset_raw".format(v)].mean()
    quaatro["{}_offset".format(v)] = quaatro[v] - quaatro["{}_certified".format(v)]

# Parse sample names
quaatro["station"] = [
    name.split(" ")[0] if name.split(" ")[0] in stations.index else ""
    for name in quaatro.name_karel
]
quaatro["bottleid"] = [
    "202000{}".format(name.split(" ")[-1])
    if "202000{}".format(name.split(" ")[-1]) in data.bottleid.to_string()
    else ""
    for name in quaatro.name_karel
]
quaatro["station_bottleid"] = [
    "{}_{}".format(*sb) if sb[0] != "" and sb[1] != "" else ""
    for sb in zip(quaatro.station, quaatro.bottleid)
]

# Merge into main data table
for v in qvars:
    data["{}_quaatro".format(v)] = np.nan
    for _, row in quaatro.iterrows():
        if row.station_bottleid in data.station_bottleid.values:
            data.loc[
                row.station_bottleid == data.station_bottleid, "{}_quaatro".format(v)
            ] = row[v]

#%% Import and process AIRICA data =====================================================
#   _____  _  ___    _  ___    _____     ___    _  ___
#  (  _  )(_)|  _`\ (_)(  _`\ (  _  )   (  _`\ (_)(  _`\
#  | (_) || || (_) )| || ( (_)| (_) |   | | ) || || ( (_)
#  |  _  || || ,  / | || |  _ |  _  |   | | | )| || |  _
#  | | | || || |\ \ | || (_( )| | | |   | |_) || || (_( )
#  (_) (_)(_)(_) (_)(_)(____/'(_) (_)   (____/'(_)(____/'

# Import AIRICA dbs file from 27 July 2021
airica = ks.airica.read_dbs("data/airica/airica-20210727/airica-20210727.dbs")
airica["bottle_split"] = airica.bottle.str.split("-")

# Assign measurement type and metadata --- CRMs
airica["crm"] = airica.bottle.str.startswith("crm")
airica["crm_batch"] = np.nan
airica["dic_certified"] = np.nan
for i, row in airica.iterrows():
    if row.crm:
        crm_batch = int(row.bottle_split[1])
        airica.loc[i, "crm_batch"] = crm_batch
        airica.loc[i, "salinity"] = 33.525
        if row.bottle != "crm-186-210315-lo":  # bad measurement
            if crm_batch == 186:
                airica.loc[i, "dic_certified"] = 2012.59

# Assign measurement type and metadata --- RWS samples
airica["rws"] = airica.bottle.str.startswith("rws")
airica["ix_data"] = np.nan
airica["station"] = ""
airica["bottleid"] = ""
airica["station_bottleid"] = ""
airica["dic_vindta"] = np.nan
for i, row in airica.iterrows():
    if row.rws:
        airica.loc[i, "station"] = row.bottle_split[1]
        airica.loc[i, "bottleid"] = row.bottle_split[2]
        station_bottleid = row.bottle_split[1] + "_" + row.bottle_split[2]
        airica.loc[i, "station_bottleid"] = station_bottleid
        ix_data = data.index[data.station_bottleid == station_bottleid][0]
        airica.loc[i, "ix_data"] = ix_data
        airica.loc[i, "salinity"] = data.loc[ix_data, "salinity"]
        airica.loc[i, "dic_vindta"] = data.loc[ix_data, "dic"]

# Re-do density conversion
airica["density"] = calk.density.seawater_1atm_MP81(
    temperature=airica.temperature, salinity=airica.salinity
)
airica["mass_sample"] = airica.volume_sample * airica.density * 1e-6  # kg

# Get CRM calibration factor and apply
airica["dic_absolute_here"] = airica.dic_certified * airica.mass_sample  # micromol
airica["calibration_factor_here"] = airica.dic_absolute_here / airica.area
airica["calibration_factor"] = airica.calibration_factor_here.mean()
airica["dic_absolute"] = airica.area * airica.calibration_factor
airica["dic"] = airica.dic_absolute / airica.mass_sample

# Put DIC values into data
data["dic_airica"] = np.nan
data.loc[airica.ix_data[~pd.isnull(airica.ix_data)], "dic_airica"] = airica.dic[
    ~pd.isnull(airica.ix_data)
].to_numpy()

# Draw VINDTA vs AIRICA
if draw_figures:
    fig, ax = plt.subplots(dpi=300, figsize=(6, 6))
    data.plot.scatter("dic", "dic_airica", ax=ax)
    ax.set_aspect(1)
    ax.grid(alpha=0.3)
    # Linear regression
    L = ~pd.isnull(data.dic) & ~pd.isnull(data.dic_airica)
    lr = linregress(data.dic[L].to_numpy(), data.dic_airica[L].to_numpy())
    # Axis settings
    ax_range = np.array(
        [
            np.min([data.dic[L].min(), data.dic_airica[L].min()]),
            np.max([data.dic[L].max(), data.dic_airica[L].max()]),
        ]
    )
    ax_extra = np.array([-10, 10])
    ax.plot(ax_range, ax_range, c="k")
    ax.plot(ax_range, ax_range * lr.slope + lr.intercept)
    ax.set_xlim(ax_range + ax_extra)
    ax.set_ylim(ax_range + ax_extra)
    ax.set_xlabel("DIC from VINDTA / micromol/kg")
    ax.set_ylabel("DIC from AIRICA / micromol/kg")

#%% Do PyCO2SYS calculations ===========================================================
#   ___           ___    _____    __    ___    _     _  ___
#  (  _`\        (  _`\ (  _  ) /'__`\ (  _`\ ( )   ( )(  _`\
#  | |_) ) _   _ | ( (_)| ( ) |(_)  ) )| (_(_)`\`\_/'/'| (_(_)
#  | ,__/'( ) ( )| |  _ | | | |   /' / `\__ \   `\ /'  `\__ \
#  | |    | (_) || (_( )| (_) | /' /( )( )_) |   | |   ( )_) |
#  (_)    `\__, |(____/'(_____)(_____/'`\____)   (_)   `\____)
#         ( )_| |
#         `\___/'

# Get sulfate from salinity to fill blanks
totals = {
    "total_sulfate": np.where(
        np.isnan(data.sulfate.values),
        pyco2.salts.sulfate_MR66(data.salinity.values) * 1e6,
        data.sulfate.values,
    ),
    "total_silicate": np.where(np.isnan(data.silicate.values), 0, data.silicate.values),
    "total_phosphate": np.where(
        np.isnan(data.phosphate.values), 0, data.phosphate.values
    ),
    "total_ammonia": np.where(np.isnan(data.ammonia.values), 0, data.ammonia.values),
}

opts_PyCO2SYS = {
    "opt_k_carbonic": opt_k_carbonic,
    "opt_total_borate": opt_total_borate,
    "opt_k_fluoride": opt_k_fluoride,
    "opt_k_bisulfate": opt_k_bisulfate,
}

# Solve from alkalinity and DIC
co2dict_alk_dic = pyco2.sys(
    data.alkalinity.values,
    data.dic.values,
    1,
    2,
    salinity=data.salinity.values,
    temperature=data.temperature.values,
    pressure=data.depth.values,
    temperature_out=25,
    pressure_out=0,
    opt_pH_scale=1,  # Total scale for consistency with spectro pH measurements
    **opts_PyCO2SYS,
    **totals,
)
data["pH_calc12_total_25"] = co2dict_alk_dic["pH_total_out"]
data["pH_calc12_total_insitu"] = co2dict_alk_dic["pH_total"]
data["fco2_calc12_insitu"] = co2dict_alk_dic["fCO2"]

# Solve from alkalinity and DIC at 20 °C
co2dict_alk_dic_20C = pyco2.sys(
    data.alkalinity.values,
    data.dic.values,
    1,
    2,
    salinity=data.salinity.values,
    temperature=20,
    opt_pH_scale=1,  # Total scale for consistency with spectro pH measurements
    **opts_PyCO2SYS,
    **totals,
)
data["pH_calc12_total_20"] = co2dict_alk_dic_20C["pH_total"]

# Convert VINDTA pH to Total scale at 25 °C
co2dict_vindta_pH = pyco2.sys(
    data.pH_vindta_free_lab.values,
    2000,
    3,
    2,
    salinity=data.salinity.values,
    temperature=data.pH_vindta_temperature.values,
    temperature_out=25,
    opt_pH_scale=3,  # VINDTA pH is on the Free scale
    **opts_PyCO2SYS,
    **totals,
)
data["pH_vindta_total_25"] = co2dict_vindta_pH["pH_total_out"]

# Convert VINDTA pH to Total scale at 20 °C
co2dict_vindta_pH_20C = pyco2.sys(
    data.pH_vindta_free_lab.values,
    2000,
    3,
    2,
    salinity=data.salinity.values,
    temperature=data.pH_vindta_temperature.values,
    temperature_out=20,
    opt_pH_scale=3,  # VINDTA pH is on the Free scale
    **opts_PyCO2SYS,
    **totals,
)
data["pH_vindta_total_20"] = co2dict_vindta_pH_20C["pH_total_out"]

# Convert spectrophotometric pH to 25 °C
co2dict_spectro_25C = pyco2.sys(
    data.pH_spectro_total_lab.values,
    2000,
    3,
    2,
    salinity=data.salinity.values,
    temperature=data.pH_spectro_temperature.values,
    temperature_out=25,
    opt_pH_scale=1,  # spectrophotometric pH is on the Total scale
    **opts_PyCO2SYS,
    **totals,
)
data["pH_spectro_total_25"] = co2dict_spectro_25C["pH_total_out"]

# Convert spectrophotometric pH to 20 °C
co2dict_spectro_20C = pyco2.sys(
    data.pH_spectro_total_lab.values,
    2000,
    3,
    2,
    salinity=data.salinity.values,
    temperature=data.pH_spectro_temperature.values,
    temperature_out=20,
    opt_pH_scale=1,  # spectrophotometric pH is on the Total scale
    **opts_PyCO2SYS,
    **totals,
)
data["pH_spectro_total_20"] = co2dict_spectro_20C["pH_total_out"]

# Convert spectrophotometric pH to in situ temperature
co2dict_spectro_insitu = pyco2.sys(
    data.pH_spectro_total_lab.values,
    2000,
    3,
    2,
    salinity=data.salinity.values,
    temperature=data.pH_spectro_temperature.values,
    temperature_out=data.temperature.values,
    pressure_out=data.depth.values,
    opt_pH_scale=1,  # spectrophotometric pH is on the Total scale
    **opts_PyCO2SYS,
    **totals,
)
data["pH_spectro_total_insitu"] = co2dict_spectro_insitu["pH_total_out"]

# Calculate other stuff from spectrophotometric pH and total alkalinity
co2dict_spectro_alkalinity = pyco2.sys(
    data.pH_spectro_total_lab.values,
    data.alkalinity.values,
    3,
    1,
    salinity=data.salinity.values,
    temperature=data.pH_spectro_temperature.values,
    temperature_out=data.temperature.values,
    pressure_out=data.depth.values,
    opt_pH_scale=1,  # spectrophotometric pH is on the Total scale
    **opts_PyCO2SYS,
    **totals,
)
data["fco2_calc13s_insitu"] = co2dict_spectro_alkalinity["fCO2_out"]

# Calculate other stuff from spectrophotometric pH and DIC
co2dict_spectro_dic = pyco2.sys(
    data.pH_spectro_total_lab.values,
    data.dic.values,
    3,
    2,
    salinity=data.salinity.values,
    temperature=data.pH_spectro_temperature.values,
    temperature_out=data.temperature.values,
    pressure_out=data.depth.values,
    opt_pH_scale=1,  # spectrophotometric pH is on the Total scale
    **opts_PyCO2SYS,
    **totals,
)
data["fco2_calc23s_insitu"] = co2dict_spectro_dic["fCO2_out"]

# Calculate other stuff from VINDTA pH and total alkalinity
co2dict_vindta_alkalinity = pyco2.sys(
    data.pH_vindta_free_lab.values,
    data.alkalinity.values,
    3,
    1,
    salinity=data.salinity.values,
    temperature=data.pH_vindta_temperature.values,
    temperature_out=data.temperature.values,
    pressure_out=data.depth.values,
    opt_pH_scale=3,  # VINDTA pH is on the Free scale
    **opts_PyCO2SYS,
    **totals,
)
data["fco2_calc13v_insitu"] = co2dict_vindta_alkalinity["fCO2_out"]

# Calculate other stuff from VINDTA pH and DIC
co2dict_vindta_dic = pyco2.sys(
    data.pH_vindta_free_lab.values,
    data.dic.values,
    3,
    2,
    salinity=data.salinity.values,
    temperature=data.pH_vindta_temperature.values,
    temperature_out=data.temperature.values,
    pressure_out=data.depth.values,
    opt_pH_scale=3,  # VINDTA pH is on the Free scale
    **opts_PyCO2SYS,
    **totals,
)
data["fco2_calc23v_insitu"] = co2dict_vindta_dic["fCO2_out"]

# Convert VINDTA pH to in situ temperature
co2dict_vindta_insitu = pyco2.sys(
    data.pH_vindta_free_lab.values,
    2000,
    3,
    2,
    salinity=data.salinity.values,
    temperature=data.pH_vindta_temperature.values,
    temperature_out=data.temperature.values,
    pressure_out=data.depth.values,
    opt_pH_scale=3,  # VINDTA pH is on the Free scale
    **opts_PyCO2SYS,
    **totals,
)
data["pH_vindta_total_insitu"] = co2dict_vindta_insitu["pH_total_out"]

#%% Assign WOCE flags ==================================================================
#   _       _  _____  ___    ___         ___  _
#  ( )  _  ( )(  _  )(  _`\ (  _`\     /'___)(_ )
#  | | ( ) | || ( ) || ( (_)| (_(_)   | (__   | |    _ _    __    ___
#  | | | | | || | | || |  _ |  _)_    | ,__)  | |  /'_` ) /'_ `\/',__)
#  | (_/ \_) || (_) || (_( )| (_( )   | |     | | ( (_| |( (_) |\__, \
#  `\___x___/'(_____)(____/'(____/'   (_)    (___)`\__,_)`\__  |(____/
#                                                        ( )_) |
#                                                         \___/'

# Initialise with generic flags
flag_vars = {
    "dic": "dic",
    "alkalinity": "alkalinity",
    "pH_spectro": "pH_spectro_total_lab",
    "pH_vindta": "pH_vindta_free_lab",
    "something": "pH_calc12_total_25",
}
for k, v in flag_vars.items():
    data[k + "_flag"] = 9
    data.loc[~np.isnan(data[v]), k + "_flag"] = 2

# ~~~ Assign custom flags: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Spectrophotometric pH
data.loc[
    np.isin(
        data.station_bottleid,
        [
            "WALCRN20_2018005589",
            "WALCRN70_2018005602",
            "NOORDWK10_2019008288",
            "NOORDWK70_2018005684",
            "NOORDWK70_2019008290",
            "TERSLG10_2018005694",
            "TERSLG10_2018005698",
            "TERSLG100_2018005719",
            "TERSLG100_2018005722",
            "TERSLG235_2019008296",
            "ROTTMPT70_2019006224",
            "ROTTMPT70_2019008298",
            "SCHOUWN10_2018005613",
            "SCHOUWN10_2019008284",
            "NOORDWK2_2018005650",
            "NOORDWK2_2019008287",
            "NOORDWK10_2019008288",
            "NOORDWK70_2018005681",
            "TERSLG50_2018005704",
            "NOORDWK10_2019006216",
            "WALCRN2_2019008281",
            "WALCRN20_2019005560",
            "ROTTMPT70_2018008778",
            "GOERE2_2020009978",
        ],
    ),
    "pH_spectro_flag",
] = 3

# Potentiometric (VINDTA) pH
data.loc[
    np.isin(
        data.station_bottleid,
        [
            "TERSLG235_2018005745",
            "WALCRN20_2019005560",
            "ROTTMPT50_2018008770",
            "ROTTMPT70_2018008778",
        ],
    ),
    "pH_vindta_flag",
] = 3

# DIC only
data.loc[
    np.isin(
        data.station_bottleid,
        [
            "WALCRN20_2019005560",
            "ROTTMPT50_2018008770",
            "ROTTMPT50_2018008766",
            "ROTTMPT50_2018008767",
            "ROTTMPT70_2018008773",
            "ROTTMPT70_2018008778",  # because it gives anomalously high fCO2
        ],
    ),
    "dic_flag",
] = 3

# Alkalinity only
data.loc[
    np.isin(
        data.station_bottleid,
        [
            "GOERE2_2020005212",
            "GOERE6_2021005974",
        ],
    ),
    "alkalinity_flag",
] = 3

# Not sure if DIC or TA is the problem here: the spectro and VINDTA pHs agree with each
# other, but not with the PyCO2SYS value.
data.loc[
    np.isin(
        data.station_bottleid,
        [
            "TERSLG100_2018005720",
            "TERSLG135_2018005732",
        ],
    ),
    ["dic_flag", "alkalinity_flag"],
] = 3
# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~

#%% Save results to file(s) ==============================================================
#   ___                                                     _    _
#  (  _`\                                                  (_ ) ( )_
#  | (_(_)   _ _  _   _    __      _ __   __    ___  _   _  | | | ,_)  ___
#  `\__ \  /'_` )( ) ( ) /'__`\   ( '__)/'__`\/',__)( ) ( ) | | | |  /',__)
#  ( )_) |( (_| || \_/ |(  ___/   | |  (  ___/\__, \| (_) | | | | |_ \__, \
#  `\____)`\__,_)`\___/'`\____)   (_)  `\____)(____/`\___/'(___)`\__)(____/

# Pickle for Python
with open("pickles/data_stations_groups_v10.pkl", "wb") as f:
    pickle.dump((data, stations, groups), f)
with open("pickles/dbs_tdata_pdata_v10.pkl", "wb") as f:
    pickle.dump((dbs, tdata, pdata), f)

# ... or, read in the above files
with open("pickles/data_stations_groups_v10.pkl", "rb") as f:
    data, stations, groups = pickle.load(f)
with open("pickles/dbs_tdata_pdata_v10.pkl", "rb") as f:
    dbs, tdata, pdata = pickle.load(f)

# Data to Excel
xlfields = [
    "datetime",
    "station",
    "bottleid",
    "latitude",
    "longitude",
    "depth",
    "temperature",
    "salinity",
    "dic",
    "dic_flag",
    "alkalinity",
    "alkalinity_flag",
    "pH_spectro_total_lab",
    "pH_spectro_temperature",
    "pH_spectro_flag",
    "pH_vindta_free_lab",
    "pH_vindta_temperature",
    "pH_vindta_flag",
    "nitrate",
    "nitrite",
    "ammonia",
    "phosphate",
    "silicate",
    "doc",
    "sulfate",
    "chloride",
]
data[xlfields].to_excel("results/data_v10.xlsx", na_rep="NaN")

# Rounded data to Excel
rdata = copy.deepcopy(data[xlfields])
rfields = {
    "latitude": 4,
    "longitude": 4,
    "dic": 1,
    "alkalinity": 1,
    "pH_spectro_total_lab": 3,
    "pH_vindta_free_lab": 3,
    "nitrate": 2,
    "nitrite": 2,
    "ammonia": 2,
    "phosphate": 2,
    "silicate": 2,
    "doc": 1,
    "sulfate": 0,
    "chloride": 0,
}
for field, decimals in rfields.items():
    rdata[field] = rdata[field].round(decimals=decimals)
rdata.to_excel("results/RWS_Noordzee_v10_raw.xlsx", na_rep="NaN")

# Export file for Onno @ Rijkswaterstaat
rws_onno = data[["station", "bottleid"]].rename(
    columns={
        "station": "Meetpunt",
        "bottleid": "Id text",
    }
)
for old, new in {
    "dic": "Dissolved_inorganic_carbon [micromol/kg]",
    "alkalinity": "Total_alkalinity [micromol/kg]",
}.items():
    rws_onno[new] = np.round(data[old], decimals=1)
rws_onno["Analysis date (DIC and TA)"] = data["analysis_datetime"].dt.strftime(
    "%d/%m/%Y"
)
rws_onno["DIC_flag"] = data.dic_flag
rws_onno["TA_flag"] = data.alkalinity_flag
rws_onno["pH_Total @ 25C"] = np.round(data["pH_spectro_total_25"], decimals=3)
rws_onno["Analysis date (pH)"] = data["pH_spectro_date_analysis"].dt.strftime(
    "%d/%m/%Y"
)
rws_onno["pH_flag"] = data.pH_spectro_flag
rws_onno.to_csv("results/RWS_NIOZ_CO2_v10.csv", index=False, na_rep="-999")

# Export file for Willem Stolte @ Deltares
rws_deltares = data[
    [
        "station",
        "bottleid",
        "latitude",
        "longitude",
        "depth",
        "datetime",
        "temperature",
        "salinity",
        "dic",
        "dic_flag",
        "alkalinity",
        "alkalinity_flag",
        "pH_spectro_total_insitu",
        "pH_spectro_flag",
    ]
].rename(
    columns={
        "bottleid": "bottle_id",
        "dic": "TCO2KG01_KGUM",
        "dic_flag": "TCO2KG01_KGUM_flag",
        "alkalinity": "MDMAP014_KGUM",
        "alkalinity_flag": "MDMAP014_KGUM_flag",
        "pH_spectro_total_insitu": "PHMASSXX",
        "pH_spectro_flag": "PHMASSXX_flag",
    }
)
rws_deltares["TCO2KG01_KGUM"] = np.round(rws_deltares.TCO2KG01_KGUM, decimals=1)
rws_deltares["MDMAP014_KGUM"] = np.round(rws_deltares.MDMAP014_KGUM, decimals=1)
rws_deltares["PHMASSXX"] = np.round(rws_deltares.PHMASSXX, decimals=3)
rws_deltares.to_csv("results/RWS_NIOZ_CO2_Deltares_v10.csv", index=False, na_rep=-999)

# Export QUASIMEME sample data
qm = dbs[dbs.bottle.str.startswith("QOA")].copy()
pyco2_qm = pyco2.sys(
    par1=qm.alkalinity,
    par2=qm.dic,
    par1_type=1,
    par2_type=2,
    salinity=qm.salinity,
    temperature=25,
    opt_k_carbonic=opt_k_carbonic,
    opt_total_borate=opt_total_borate,
    opt_k_bisulfate=opt_k_bisulfate,
    opt_k_fluoride=opt_k_fluoride,
)
qm["pH_calc12_free_lab"] = pyco2_qm["pH_free"]
qm = qm[
    [
        "bottle",
        "salinity",
        "alkalinity",
        "dic",
        # "pH_vindta_free_lab",
        # "pH_calc12_free_lab",
    ]
]
qm["alkalinity"] = np.round(qm.alkalinity, decimals=1)
qm["dic"] = np.round(qm.dic, decimals=1)
qm.to_csv("results/QUASIMEME_CO2_NIOZ.csv", index=False)
