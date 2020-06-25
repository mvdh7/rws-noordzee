import pickle
import numpy as np
import pandas as pd
from matplotlib import dates as mdates

# from sys import path

# kspath = "/home/matthew/github/koolstof"
# if kspath not in path:
#     path.append(kspath)
import koolstof as ks

# calkpath = "/home/matthew/github/calkulate"
# if calkpath not in path:
#     path.append(calkpath)
import calkulate as calk

# Import station positions
allstations = pd.read_excel("data/Coordinaten_verzuring_20190429.xlsx")
allstations["lat"] = (
    allstations.lat_deg + allstations.lat_min / 60 + allstations.lat_sec / 3600
)
allstations["lon"] = (
    allstations.lon_deg + allstations.lon_min / 60 + allstations.lon_sec / 3600
)

# Import CO2 system measurements and add station positions
dataversion = "20200406"
data = pd.read_excel(
    "data/co2data-{}.xlsx".format(dataversion),
    sheet_name="2018_2019 All Final Data",
    header=2,
    na_values=["#N/B", -999],
)
withdata = np.array([isinstance(station, str) for station in data.station.values])
data = data[withdata].reset_index()
data["lat"] = np.nan
data["lon"] = np.nan
for S in range(len(data.index)):
    SL = allstations["U meetpunt"] == data.station[S]
    if np.any(SL):
        data.loc[S, "lat"] = allstations.lat[SL].values[0]
        data.loc[S, "lon"] = allstations.lon[SL].values[0]
    else:
        print("Warning: station {} has no co-ordinates!".format(data.station[S]))
data["depth"] = 3.5

# Import bottle file metadata and merge into data
meta = pd.read_excel(
    "data/Bottlefile_NIOZ_20191105_MPH.xlsx", header=3, na_values=["#N/B", -999]
)
metavars = [
    "temperature",
    "nitrite_gvol",
    "nitrate_gvol",
    "ammonia_gvol",
    "phosphate_gvol",
    "silicate_gvol",
    "doc_gvol",
    "pH_rws_total_insitu",
    "salinity_rws",
    "chloride_gvol",
    "sulfate_gvol",
]
for metavar in metavars:
    data[metavar] = np.nan
    for D in range(len(data.index)):
        DL = data.bottleid[D] == meta.bottleid
        if any(DL):
            data.loc[D, metavar] = meta[metavar][DL].values[0]

# Finalise salinity
data["salinity"] = data.salinity_rws
for D in range(len(data.index)):
    if np.isnan(data.loc[D, "salinity"]):
        data.loc[D, "salinity"] = data.salinity_nioz[D]

# Convert units from mg/l to μmol/l
data["doc_vol"] = 1e3 * data.doc_gvol / ks.molar.mass["C"]
data["nitrate_vol"] = 1e3 * data.nitrate_gvol / ks.molar.mass["N"]
data["nitrite_vol"] = 1e3 * data.nitrite_gvol / ks.molar.mass["N"]
data["ammonia_vol"] = 1e3 * data.ammonia_gvol / ks.molar.mass["N"]
data["din_vol"] = data.nitrate_vol + data.nitrite_vol + data.ammonia_vol
data["silicate_vol"] = 1e3 * data.silicate_gvol / ks.molar.mass["Si"]
data["phosphate_vol"] = (
    data.phosphate_gvol / ks.molar.mass["P"]
)  # provided in μg/l, not mg/l
data["sulfate_vol"] = 1e3 * data.sulfate_gvol / ks.molar.mass["SO4"]
data["chloride_vol"] = 1e3 * data.chloride_gvol / ks.molar.mass["Cl"]

# Convert to molinity (μmol/kg-sw)
nutrients_temperature = 23  # presumed - check! <-------------------------------- TO DO
data["density_nutrients"] = calk.density.seawater_atm_MP81(
    nutrients_temperature, data.salinity
)
nutrients = [
    "doc",
    "nitrate",
    "nitrite",
    "ammonia",
    "din",
    "silicate",
    "phosphate",
    "sulfate",
    "chloride",
]
for nutrient in nutrients:
    data[nutrient] = data[nutrient + "_vol"] / data.density_nutrients
    data.loc[data[nutrient] < 0, nutrient] = 0  # set negative values to zero

# Nutrient uncertainties from RWS
data["silicate_unc"] = data.silicate * 0.15
data["phosphate_unc"] = data.phosphate * 0.25
data["nitrate_unc"] = data.nitrate * 0.3
data["nitrite_unc"] = data.nitrite * 0.3
data["ammonia_unc"] = data.ammonia * 0.15
data["doc_unc"] = data.doc * 0.2

# Get unique stations table and assign properties
stations = pd.DataFrame(index=np.unique(data.station))
stations["lat"] = np.nan
stations["lon"] = np.nan
for station in stations.index:
    SL = data.station == station
    stations.loc[station, "lat"] = np.unique(data.lat[SL])
    stations.loc[station, "lon"] = np.unique(data.lon[SL])
stations["r"] = np.nan
stations["g"] = np.nan
stations["b"] = np.nan
# Goeree in red:
stations.loc["GOERE2", ["r", "g", "b"]] = np.array([0.9, 0.1, 0.1]) * 1.11
stations.loc["GOERE6", ["r", "g", "b"]] = np.array([0.9, 0.1, 0.1]) * 0.8
# Noordwijk in purple:
stations.loc["NOORDWK2", ["r", "g", "b"]] = np.array([0.6, 0.3, 0.6]) * 1.4
stations.loc["NOORDWK10", ["r", "g", "b"]] = np.array([0.6, 0.3, 0.6]) * 1.1
stations.loc["NOORDWK20", ["r", "g", "b"]] = np.array([0.6, 0.3, 0.6]) * 0.9
stations.loc["NOORDWK70", ["r", "g", "b"]] = np.array([0.6, 0.3, 0.6]) * 0.6
# Rottumerplaat in green:
stations.loc["ROTTMPT50", ["r", "g", "b"]] = np.array([0.3, 0.7, 0.3]) * 1.1
stations.loc["ROTTMPT70", ["r", "g", "b"]] = np.array([0.3, 0.7, 0.3]) * 0.7
# Schouwen in orange:
stations.loc["SCHOUWN10", ["r", "g", "b"]] = np.array([1, 0.5, 0])
# Terschelling in blue:
stations.loc["TERSLG10", ["r", "g", "b"]] = np.array([0.2, 0.5, 0.7]) * 1.42
stations.loc["TERSLG50", ["r", "g", "b"]] = np.array([0.2, 0.5, 0.7]) * 1.2
stations.loc["TERSLG100", ["r", "g", "b"]] = np.array([0.2, 0.5, 0.7]) * 1.0
stations.loc["TERSLG135", ["r", "g", "b"]] = np.array([0.2, 0.5, 0.7]) * 0.8
stations.loc["TERSLG175", ["r", "g", "b"]] = np.array([0.2, 0.5, 0.7]) * 0.6
stations.loc["TERSLG235", ["r", "g", "b"]] = np.array([0.2, 0.5, 0.7]) * 0.3
# Walcheren in brown:
stations.loc["WALCRN2", ["r", "g", "b"]] = np.array([0.6, 0.3, 0.1]) * 1.2
stations.loc["WALCRN20", ["r", "g", "b"]] = np.array([0.6, 0.3, 0.1]) * 0.9
stations.loc["WALCRN70", ["r", "g", "b"]] = np.array([0.6, 0.3, 0.1]) * 0.6
# Merge into rgb
stations_rgb = stations[["r", "g", "b"]].values.tolist()
stations_rgb = [np.array([v]) for v in stations_rgb]
stations["rgb"] = stations_rgb

# Assign groups
groups = pd.DataFrame(
    index=[
        "Walcheren & Schouwen",
        "Goeree",
        "Noordwijk",
        "Terschelling & Rottumerplaat",
    ]
)
groups["gid"] = [1, 2, 3, 4]
stations["gid"] = 0
for station in stations.index:
    if station in ["WALCRN2", "WALCRN20", "WALCRN70", "SCHOUWN10"]:
        stations.loc[station, "gid"] = 1
    elif station in ["GOERE2", "GOERE6"]:
        stations.loc[station, "gid"] = 2
    elif station in ["NOORDWK2", "NOORDWK10", "NOORDWK20", "NOORDWK70"]:
        stations.loc[station, "gid"] = 3
    elif station in [
        "TERSLG10",
        "TERSLG50",
        "TERSLG100",
        "TERSLG135",
        "TERSLG175",
        "TERSLG235",
        "ROTTMPT50",
        "ROTTMPT70",
    ]:
        stations.loc[station, "gid"] = 4
data["gid"] = stations.gid[data.station].values
data["rgb"] = stations.rgb[data.station].values

# Add date stuff
data["datenum"] = mdates.date2num(data.datetime)
data["day_of_year"] = data.datetime.dt.dayofyear

# Save for making figures in MATLAB
data.to_csv("pickles/co2data-{}.csv".format(dataversion), index=False)
stations.to_csv("pickles/stations-{}.csv".format(dataversion), index_label="station")
groups.to_csv("pickles/groups-{}.csv".format(dataversion), index_label="group")

# Save for next step of Python analysis
with open("pickles/data0-{}.pkl".format(dataversion), "wb") as f:
    pickle.dump((data, stations, groups), f)
