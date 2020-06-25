#%% ====================================================================================
#   ___                _
#  (  _`\             (_ )
#  | |_) ) _ __   _    | |    _      __   _   _    __
#  | ,__/'( '__)/'_`\  | |  /'_`\  /'_ `\( ) ( ) /'__`\
#  | |    | |  ( (_) ) | | ( (_) )( (_) || (_) |(  ___/
#  (_)    (_)  `\___/'(___)`\___/'`\__  |`\___/'`\____)
#                                 ( )_) |
#                                  \___/'

import copy, pickle, re
import numpy as np
import pandas as pd
import PyCO2SYS as pyco2
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import dates as mdates

# ~~~ Set up imports from path: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# from sys import path

# kspath = "/home/matthew/github/koolstof"
# if kspath not in path:
#     path.append(kspath)
import koolstof as ks

kvi = ks.vindta.io
kvp = ks.vindta.process

# calkpath = "/home/matthew/github/calkulate"
# if calkpath not in path:
#     path.append(calkpath)
import calkulate as calk

# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~

#%% ====================================================================================
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
draw_figures = False  # draw figures?
# save_figures = False  # save figures to file? (if draw_figures = True)

# PyCO2SYS options, also used by Calkulate
K1K2_opt = 10
KSO4_BSal_opt = 3
KF_opt = 2

# Calkulate options
temperature_override = None  # set to None to ignore; ~23 makes the pH work very well

# Poison corrections for DIC and alkalinity
sample_volume = 250e-3  # sample bottles in ml
poison_volume = 250e-6  # added HgCl2 in ml
# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~

# Get new-style PyCO2SYS settings inputs for Calkulate
KSO4_opt, BSal_opt = pyco2.convert.options_old2new(KSO4_BSal_opt)
KSO4_opt = KSO4_opt[0]
BSal_opt = BSal_opt[0]

# Matplotlib universal settings
mpl.rcParams["font.family"] = "Open Sans"
mpl.rcParams["font.size"] = 10

#%% ====================================================================================
#   _   _  _  _   _  ___   _____  _____       ___    _
#  ( ) ( )(_)( ) ( )(  _`\(_   _)(  _  )    /'___)_ (_ )
#  | | | || || `\| || | ) | | |  | (_) |   | (__ (_) | |    __    ___
#  | | | || || , ` || | | ) | |  |  _  |   | ,__)| | | |  /'__`\/',__)
#  | \_/ || || |`\ || |_) | | |  | | | |   | |   | | | | (  ___/\__, \
#  `\___/'(_)(_) (_)(____/' (_)  (_) (_)   (_)   (_)(___)`\____)(____/

# ~~~ File import settings: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
vindta_path = "data/VINDTA#15 Data/"  # path to the logfile and .dbs files
logfile_fname = "logfile_20200407.bak"  # the logfile's filename
dbs_fnames = [  # list of .dbs filenames
    "2018/2018_{}_RWS_CO2.dbs".format(month) for month in ["June", "Aug", "Nov"]
] + ["2019/2019_{}_RWS_CO2.dbs".format(month) for month in ["02", "08", "11"]]
methods = [  # list of VINDTA methods that are considered as measurements
    "3C standard",
    "3C standardRWS",
]
# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~

# Import VINDTA logfile and .dbs file(s)
logfile = kvi.read_logfile(vindta_path + logfile_fname, methods=methods)
dbs = pd.concat([kvi.read_dbs(vindta_path + fname) for fname in dbs_fnames])
dbs.reset_index(drop=True, inplace=True)

# Set up table of DIC analysis sessions
dic_sessions = dbs.groupby(by="cell ID")["cell ID"].agg(count="count")
dic_sessions["analysis_datenum"] = dbs.groupby(by="cell ID").analysis_datenum.agg(
    lambda x: np.median(x)
)
dic_sessions["analysis_datetime"] = mdates.num2date(dic_sessions.analysis_datenum)

# Get index for logfile rows corresponding to the dbs table
dbs["logfile_index"] = dbs.apply(kvi.get_logfile_index, args=[logfile], axis=1)

#%% ====================================================================================
#   ___           _                         _               _         _
#  (  _`\        ( )_                      ( )_            ( )       ( )_
#  | ( (_)   __  | ,_)     ___ ___     __  | ,_)   _ _    _| |   _ _ | ,_)   _ _
#  | |___  /'__`\| |     /' _ ` _ `\ /'__`\| |   /'_` ) /'_` | /'_` )| |   /'_` )
#  | (_, )(  ___/| |_    | ( ) ( ) |(  ___/| |_ ( (_| |( (_| |( (_| || |_ ( (_| |
#  (____/'`\____)`\__)   (_) (_) (_)`\____)`\__)`\__,_)`\__,_)`\__,_)`\__)`\__,_)

# ~~~ Import data and stations tables: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dataversion = "20200406"
with open("pickles/data0-{}.pkl".format(dataversion), "rb") as f:
    data, stations, groups = pickle.load(f)

# Rename columns to avoid clashes
data.rename(mapper={"dic": "dic_original"}, axis=1, inplace=True)

# Define dict of columns of data (keys) to transfer to dbs (values) below
data2dbs = {
    "salinity": "salinity",
    "silicate": "silicate",
    "phosphate": "phosphate",
    "sulfate": "sulfate",
    "ammonia": "ammonia",
}
# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~

# Get indices of data in dbs and transfer metadata across with it (with overwriting)
dbs["data_index"] = dbs.apply(kvp.get_data_index, args=[data], axis=1)
for data_field, dbs_field in data2dbs.items():
    if dbs_field not in dbs.columns:
        dbs[dbs_field] = np.nan
    L_dbs = ~np.isnan(dbs.data_index)
    dbs.loc[L_dbs, dbs_field] = data.loc[dbs.loc[L_dbs].data_index][data_field].values

#%% ====================================================================================
#   ___    _  ___       _      _                  _
#  (  _`\ (_)(  _`\    ( )    (_ )               ( )
#  | | ) || || ( (_)   | |_    | |    _ _   ___  | |/')   ___
#  | | | )| || |  _    | '_`\  | |  /'_` )/' _ `\| , <  /',__)
#  | |_) || || (_( )   | |_) ) | | ( (_| || ( ) || |\`\ \__, \
#  (____/'(_)(____/'   (_,__/'(___)`\__,_)(_) (_)(_) (_)(____/

# ~~~ Define when to start measuring blank: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
use_from = 6  # runtime (in minutes) after which blank is measured
# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~

# Get sample-by-sample blanks
dbs["blank_here"] = dbs.apply(
    kvp.get_sample_blanks, args=[logfile], axis=1, use_from=use_from
)

# ~~~ Define which blanks are reliable: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dbs["blank_good"] = dbs.blank_here < 1000
# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~

# Get session blanks and estimate for each sample, overwriting existing "blank" column
dic_sessions = dic_sessions.join(
    dbs.groupby(by="cell ID").apply(kvp.get_session_blanks)
)
dbs = kvp.apply_session_blanks(dbs, dic_sessions)

# Plot DIC coulometer increments and blanks
if draw_figures:
    ks.vindta.plot.increments(dbs, logfile, use_from)
    for dic_session in dic_sessions.index:
        ks.vindta.plot.blanks(
            dbs[dbs["cell ID"] == dic_session], dic_sessions, title=dic_session
        )

#%% ====================================================================================
#   ___           _       _                  _              ___    _  ___
#  (  _`\        (_ )  _ ( )                ( )_           (  _`\ (_)(  _`\
#  | ( (_)   _ _  | | (_)| |_    _ __   _ _ | ,_)   __     | | ) || || ( (_)
#  | |  _  /'_` ) | | | || '_`\ ( '__)/'_` )| |   /'__`\   | | | )| || |  _
#  | (_( )( (_| | | | | || |_) )| |  ( (_| || |_ (  ___/   | |_) || || (_( )
#  (____/'`\__,_)(___)(_)(_,__/'(_)  `\__,_)`\__)`\____)   (____/'(_)(____/'

# ~~~ Identify CRMs and their batch numbers: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dbs["crm_batch"] = np.nan
dbs.loc[dbs.station == 666, "crm_batch"] = 171

# DIC analysis settings
dic_analysis_temperature = 23
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
dbs["density_analysis"] = calk.density.seawater_atm_MP81(
    dic_analysis_temperature, dbs.salinity
)

# Get CRM calibration factors
dbs["dic_calibration_factor"] = ks.crm.dic_calibration_factor(data=dbs)

# ~~~ QC the CRM calibration factors: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dbs["dic_calibration_good"] = True
dbs.loc[
    (dbs.dic_calibration_factor < 0.011)
    & (dbs.dbs_fname == vindta_path + "2018/2018_Aug_RWS_CO2.dbs"),
    "dic_calibration_good",
] = False
dbs.loc[
    (dbs.dic_calibration_factor > 0.0114)
    & (dbs.dbs_fname == vindta_path + "2019/2019_02_RWS_CO2.dbs"),
    "dic_calibration_good",
] = False
dbs.loc[
    (dbs.dic_calibration_factor < 0.01104)
    & (dbs.dbs_fname == vindta_path + "2019/2019_11_RWS_CO2.dbs"),
    "dic_calibration_good",
] = False
# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~

# Get session-averaged calibration factors
dgcf = dbs.loc[dbs.dic_calibration_good].groupby(by="cell ID").dic_calibration_factor
dic_sessions["dic_calibration_mean"] = dgcf.agg("mean")
dic_sessions["dic_calibration_std"] = dgcf.agg("std")
dic_sessions["dic_calibration_count"] = dgcf.agg(lambda x: np.sum(~np.isnan(x)))

# Move session calibration factors into dbs and calibrate everything
dbs["dic_calibration_session"] = np.nan
dbs["dic"] = np.nan
for session in dic_sessions.index:
    S = dbs["cell ID"] == session
    dbs.loc[S, "dic_calibration_session"] = dic_sessions.loc[
        session
    ].dic_calibration_mean
dbs.loc[:, "dic"] = ks.crm.calibrate_dic(
    data=dbs, calibration_factor="dic_calibration_session"
)
dbs["dic_certified_offset"] = dbs.dic - dbs.dic_certified

# ~~~ QC DIC measurements: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dbs["dic_good"] = True
dbs.loc[
    (dbs.bottle == "ROTTMPT70_2019006224") & (dbs.counts < 15000), "dic_good"
] = False
dbs.loc[dbs.bottle == "WALCRN2_2018005578", "dic_good"] = False
dbs.loc[dbs.bottle == "TERSLG235_2018005745", "dic_good"] = False
# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~

# Put good DIC results back into data table, averaging duplicates
data = kvp.dbs2data(dbs, data, "dic")

# Plot CRM calibration factors through time
if draw_figures:
    only_good = False
    if only_good:
        dbs_plot = dbs[dbs.dic_calibration_good]
    else:
        dbs_plot = dbs
    ax = ks.vindta.plot.dic_calibration_factors(dbs_plot, dic_sessions)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m"))
    ax.set_xlabel("Month of year of analysis")
    ax.set_title("All RWS analysis")
    for dbs_fname in dbs_fnames:
        dbs_subset = dbs_plot[dbs_plot.dbs_fname == vindta_path + dbs_fname]
        if np.size(dbs_subset) > 0:
            ax = ks.vindta.plot.dic_calibration_factors(dbs_subset, dic_sessions)
            ax.xaxis.set_major_locator(mdates.DayLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
            ax.set_xlabel("Day of month of analysis")
            ax.set_title(dbs_fname)

# Apply mercuric chloride correction to DIC in data (but not in dbs)
data["dic_poisoned"] = copy.deepcopy(data.dic)
data["dic"] = kvp.poison_correction(data.dic_poisoned, sample_volume, poison_volume)

#%% ====================================================================================
#   ___           _    _             _           _              _____  _____
#  (  _`\        (_ ) ( )           (_ )        ( )_           (_   _)(  _  )
#  | ( (_)   _ _  | | | |/')  _   _  | |    _ _ | ,_)   __       | |  | (_) |
#  | |  _  /'_` ) | | | , <  ( ) ( ) | |  /'_` )| |   /'__`\     | |  |  _  |
#  | (_( )( (_| | | | | |\`\ | (_) | | | ( (_| || |_ (  ___/     | |  | | | |
#  (____/'`\__,_)(___)(_) (_)`\___/'(___)`\__,_)`\__)`\____)     (_)  (_) (_)

# Set up titration table (tdata) for Calkulate
dbs2calkulate = {  # keys = dbs column names, values = titration table column names
    "bottle": "file_name",  # needs more processing after this
    "salinity": "salinity",
    "alkalinity_certified": "alkalinity_certified",
    "dbs_fname": "analysis_batch",
    "ammonia": "total_ammonia",
    "dic": "total_carbonate",
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
tdata["bisulfate_constant"] = KSO4_opt
tdata["borate_ratio"] = BSal_opt
tdata["carbonic_constants"] = K1K2_opt
tdata["fluoride_constant"] = KF_opt

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

# CRMs to not use for calibration
tdata["reference_good"] = True
tdata.loc[
    np.isin(dbs.bottle, ["CRM#171-0745", "CRM#171-0745B"]), "reference_good"
] = False
# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~

# Convert tdata to Calkulate titration dataset, calibrate and solve and return to dbs
tdata = calk.Dataset(tdata)
tdata.calibrate_and_solve()
dbs["alkalinity"] = tdata.table.alkalinity
dbs["emf0"] = tdata.table.emf0
dbs["pH_vindta_free_lab"] = tdata.table.pH
dbs["pH_vindta_temperature"] = tdata.table.pH_temperature

# ~~~ QC alkalinity measurements: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dbs["alkalinity_good"] = True
dbs.loc[
    (dbs.bottle == "NOORDWK20") & (dbs["cell ID"] == "C_Aug12-19_0708"),
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
data = kvp.dbs2data(
    dbs, data, ["alkalinity", "emf0", "pH_vindta_free_lab", "pH_vindta_temperature"]
)

# Apply mercuric chloride correction to alkalinity in data (but not in dbs or tdata)
data["alkalinity_poisoned"] = copy.deepcopy(data.alkalinity)
data["alkalinity"] = kvp.poison_correction(
    data.alkalinity_poisoned, sample_volume, poison_volume
)

# Plot calibrated acid molinities, all batches together
if draw_figures:
    tdata.plot_calibration()

# ~~~ Plot each batch's acid calibration separately: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if draw_figures:
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(12, 6))
    axs = axs.ravel()
    for i, ax in enumerate(axs):
        tdata.plot_calibration(ax=ax, batches=tdata.batches.index[i])
    for ax in axs:
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
        ax.set_xlabel("Day of month")
    plt.tight_layout()
# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~

#%% ====================================================================================
#   ___                         _                          _   _
#  (  _`\                      ( )_                       ( ) ( )
#  | (_(_) _ _      __     ___ | ,_) _ __   _       _ _   | |_| |
#  `\__ \ ( '_`\  /'__`\ /'___)| |  ( '__)/'_`\    ( '_`\ |  _  |
#  ( )_) || (_) )(  ___/( (___ | |_ | |  ( (_) )   | (_) )| | | |
#  `\____)| ,__/'`\____)`\____)`\__)(_)  `\___/'   | ,__/'(_) (_)
#         | |                                      | |
#         (_)                                      (_)

# Define pH file names (for Excel files prepared by Sharyn/Karel)
pH_root = "data/pH Spec/pH Data "
pH_files = [
    pH_root + pH_file
    for pH_file in [
        "2018/2018_JUNE_RWS_PH/2018_JUNE_RWS_PH_JAN_FEB_MAR_APR.xlsx",
        "2018/2018_AUG_RWS_PH/2018_AUG_RWS_PH_MAY_JUN_JUL_MPH.xlsx",
        "2018/2018_Nov_RWS_pH/2018_NOV_RWS_PH_SEP_OCT.xlsx",
        "2019/2019_02_RWS_pH/2019_02_RWS_pH_Results_MPH.xlsx",
        "2019/2019_08_RWS_Ph_with Correct Salinities/2019_08_RWS_Ph_Results_MPH.xlsx",
        "2019/2019_11_RWS_pH/2019_11_RWS_pH_MPH.xlsx",
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
keep_cols = list(set(rename_mapper.values())) + ["fname"]

# ~~~ Import spectrophotometric pH data from lab: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import spectro pH data #0: June 2018
pdata0 = pd.read_excel(pH_files[0], skiprows=list(range(17)) + [18], nrows=272)
pdata0.rename(mapper=rename_mapper, axis=1, inplace=True)
pdata0["fname"] = pH_files[0]

# Import spectro pH data #1: August 2018
pdata1 = pd.read_excel(pH_files[1], skiprows=list(range(16)) + [17], nrows=231)
pdata1.rename(mapper=rename_mapper, axis=1, inplace=True)
pdata1["fname"] = pH_files[1]

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
# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~

# Join all files together and recalculate pH
pdata = pd.concat(
    [pdata[keep_cols] for pdata in [pdata0, pdata1, pdata2, pdata3, pdata4, pdata5]]
)
pdata.reset_index(drop=True, inplace=True)


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
pdata["is_sample"] = pdata.station_bottleid.apply(
    lambda x: str(x).split("_")[0] in stations.index
)

# ~~~ QC lab pH measurements: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pdata["pH_good"] = True
pdata.loc[
    (pdata.station_bottleid == "TERSLG10_2019007106") & (pdata.salinity_lab == 1),
    "pH_good",
] = False
# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~

# Aggregate good pH measurements
pH_spectro_total_lab = (
    pdata[pdata.is_sample & pdata.pH_good]
    .groupby("station_bottleid")
    .pH_spectro_total_lab.agg([np.mean, np.std, np.size])
    .rename(mapper={"mean": "pH_mean", "std": "pH_std"}, axis=1)
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


# Put results into data
data = pd.concat([data, data.apply(get_pH_spectro, axis=1)], axis=1)

#%% ====================================================================================
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
    "TSO4": np.where(
        np.isnan(data.sulfate.values),
        pyco2.salts.sulfate_MR66(data.salinity.values) * 1e6,
        data.sulfate.values,
    )
}

# Solve from alkalinity and DIC
co2dict_alk_dic = pyco2.CO2SYS(
    data.alkalinity.values,
    data.dic.values,
    1,
    2,
    data.salinity.values,
    25,
    data.temperature.values,
    0,
    0,  # haven't got in situ pressures?
    data.silicate.values,
    data.phosphate.values,
    1,  # Total scale for consistency with spectrophotometric pH measurements
    K1K2_opt,
    KSO4_BSal_opt,
    KFCONSTANT=KF_opt,
    NH3=data.ammonia.values,
    totals=totals,
)
data["pH_calc12_total_25"] = co2dict_alk_dic["pHinTOTAL"]
data["pH_calc12_total_insitu"] = co2dict_alk_dic["pHoutTOTAL"]
data["fco2_calc12_insitu"] = co2dict_alk_dic["fCO2out"]

# Convert VINDTA pH to Total scale at 25 °C
co2dict_vindta_pH = pyco2.CO2SYS(
    data.pH_vindta_free_lab.values,
    2000,
    3,
    2,
    data.salinity.values,
    data.pH_vindta_temperature.values,
    25,
    0,
    0,
    data.silicate.values,
    data.phosphate.values,
    3,  # VINDTA pH is on the Free scale
    K1K2_opt,
    KSO4_BSal_opt,
    KFCONSTANT=KF_opt,
    NH3=data.ammonia.values,
    totals=totals,
)
data["pH_vindta_total_25"] = co2dict_vindta_pH["pHoutTOTAL"]

# Convert spectrophotometric pH to 25 °C
co2dict_spectro_insitu = pyco2.CO2SYS(
    data.pH_spectro_total_lab.values,
    2000,
    3,
    2,
    data.salinity.values,
    data.pH_spectro_temperature.values,
    25,
    0,
    0,
    data.silicate.values,
    data.phosphate.values,
    1,  # spectrophotometric pH is on the Total scale
    K1K2_opt,
    KSO4_BSal_opt,
    KFCONSTANT=KF_opt,
    NH3=data.ammonia.values,
    totals=totals,
)
data["pH_spectro_total_25"] = co2dict_spectro_insitu["pHoutTOTAL"]

# Convert spectrophotometric pH to in situ temperature
co2dict_spectro_insitu = pyco2.CO2SYS(
    data.pH_spectro_total_lab.values,
    2000,
    3,
    2,
    data.salinity.values,
    data.pH_spectro_temperature.values,
    data.temperature.values,
    0,
    0,
    data.silicate.values,
    data.phosphate.values,
    1,  # spectrophotometric pH is on the Total scale
    K1K2_opt,
    KSO4_BSal_opt,
    KFCONSTANT=KF_opt,
    NH3=data.ammonia.values,
    totals=totals,
)
data["pH_spectro_total_insitu"] = co2dict_spectro_insitu["pHoutTOTAL"]

# Calculate other stuff from spectrophotometric pH and total alkalinity
co2dict_spectro_alkalinity = pyco2.CO2SYS(
    data.pH_spectro_total_lab.values,
    data.alkalinity.values,
    3,
    1,
    data.salinity.values,
    data.pH_spectro_temperature.values,
    data.temperature.values,
    0,
    0,
    data.silicate.values,
    data.phosphate.values,
    1,  # spectrophotometric pH is on the Total scale
    K1K2_opt,
    KSO4_BSal_opt,
    KFCONSTANT=KF_opt,
    NH3=data.ammonia.values,
    totals=totals,
)
data["fco2_calc13s_insitu"] = co2dict_spectro_alkalinity["fCO2out"]

# Calculate other stuff from spectrophotometric pH and DIC
co2dict_spectro_dic = pyco2.CO2SYS(
    data.pH_spectro_total_lab.values,
    data.dic.values,
    3,
    2,
    data.salinity.values,
    data.pH_spectro_temperature.values,
    data.temperature.values,
    0,
    0,
    data.silicate.values,
    data.phosphate.values,
    1,  # spectrophotometric pH is on the Total scale
    K1K2_opt,
    KSO4_BSal_opt,
    KFCONSTANT=KF_opt,
    NH3=data.ammonia.values,
    totals=totals,
)
data["fco2_calc23s_insitu"] = co2dict_spectro_dic["fCO2out"]

# Calculate other stuff from VINDTA pH and total alkalinity
co2dict_vindta_alkalinity = pyco2.CO2SYS(
    data.pH_vindta_free_lab.values,
    data.alkalinity.values,
    3,
    1,
    data.salinity.values,
    data.pH_vindta_temperature.values,
    data.temperature.values,
    0,
    0,
    data.silicate.values,
    data.phosphate.values,
    3,  # VINDTA pH is on the Free scale
    K1K2_opt,
    KSO4_BSal_opt,
    KFCONSTANT=KF_opt,
    NH3=data.ammonia.values,
    totals=totals,
)
data["fco2_calc13v_insitu"] = co2dict_vindta_alkalinity["fCO2out"]

# Calculate other stuff from VINDTA pH and DIC
co2dict_vindta_dic = pyco2.CO2SYS(
    data.pH_vindta_free_lab.values,
    data.dic.values,
    3,
    2,
    data.salinity.values,
    data.pH_vindta_temperature.values,
    data.temperature.values,
    0,
    0,
    data.silicate.values,
    data.phosphate.values,
    3,  # VINDTA pH is on the Free scale
    K1K2_opt,
    KSO4_BSal_opt,
    KFCONSTANT=KF_opt,
    NH3=data.ammonia.values,
    totals=totals,
)
data["fco2_calc23v_insitu"] = co2dict_vindta_dic["fCO2out"]

# Convert VINDTA pH to in situ temperature
co2dict_vindta_insitu = pyco2.CO2SYS(
    data.pH_vindta_free_lab.values,
    2000,
    3,
    2,
    data.salinity.values,
    data.pH_vindta_temperature.values,
    data.temperature.values,
    0,
    0,
    data.silicate.values,
    data.phosphate.values,
    3,  # VINDTA pH is on the Free scale
    K1K2_opt,
    KSO4_BSal_opt,
    KFCONSTANT=KF_opt,
    NH3=data.ammonia.values,
    totals=totals,
)
data["pH_vindta_total_insitu"] = co2dict_vindta_insitu["pHoutTOTAL"]

#%% ====================================================================================
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
        ],
    ),
    "pH_spectro_flag",
] = 3

# Potentiometric (VINDTA) pH
data.loc[data.station_bottleid == "TERSLG235_2018005745", "pH_vindta_flag"] = 3

# Not sure if DIC or TA is the problem here: the spectro and VINDTA pHs agree with each
# other, but not with the PyCO2SYS value.
data.loc[
    np.isin(data.station_bottleid, ["TERSLG100_2018005720", "TERSLG135_2018005732",],),
    ["dic_flag", "alkalinity_flag"],
] = 3
# ~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~~~^~~~~

#%% ====================================================================================
#   ___                                                     _    _
#  (  _`\                                                  (_ ) ( )_
#  | (_(_)   _ _  _   _    __      _ __   __    ___  _   _  | | | ,_)  ___
#  `\__ \  /'_` )( ) ( ) /'__`\   ( '__)/'__`\/',__)( ) ( ) | | | |  /',__)
#  ( )_) |( (_| || \_/ |(  ___/   | |  (  ___/\__, \| (_) | | | | |_ \__, \
#  `\____)`\__,_)`\___/'`\____)   (_)  `\____)(____/`\___/'(___)`\__)(____/

# Pickle for Python
with open("pickles/data1-{}.pkl".format(dataversion), "wb") as f:
    pickle.dump((data, stations, groups, dbs, tdata, pdata), f)

# Data to Excel
xlfields = [
    "datetime",
    "station",
    "bottleid",
    "lat",
    "lon",
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
data[xlfields].to_excel("results/data_2018-2019.xlsx", na_rep="NaN")

#%% Rounded data to Excel
rdata = copy.deepcopy(data[xlfields])
rfields = {
    "lat": 4,
    "lon": 4,
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
rdata.to_excel("results/RWS_Noordzee_v6_raw.xlsx", na_rep="NaN")
