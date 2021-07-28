import pickle
import numpy as np, pandas as pd, matplotlib as mpl
from matplotlib import dates as mdates

# # Use environment rws_dev
# from sys import path

# for extra in ["C:/Users/mphum/GitHub/koolstof", "C:/Users/mphum/GitHub/calkulate"]:
#     if extra not in path:
#         path.append(extra)

import koolstof as ks

mpl.rcParams["date.epoch"] = "1970-01-01T00:00:00"

#%% Import station positions
allstations = pd.read_excel("data/Coordinaten_verzuring_20190429.xlsx")
allstations["latitude"] = (
    allstations.lat_deg + allstations.lat_min / 60 + allstations.lat_sec / 3600
)
allstations["longitude"] = (
    allstations.lon_deg + allstations.lon_min / 60 + allstations.lon_sec / 3600
)

# Import RWS bottle file
data = pd.read_csv(
    "data/bottle_files/Bottlefile_NIOZ_20210618_MPH.csv",
    header=3,
    na_values=["#N/B", -999],
)

#%% Assign station locations and bottle IDs
data["latitude"] = np.nan
data["longitude"] = np.nan
for S in range(len(data.index)):
    SL = allstations["U meetpunt"] == data.station[S]
    if np.any(SL):
        data.loc[S, "latitude"] = allstations.latitude[SL].values[0]
        data.loc[S, "longitude"] = allstations.longitude[SL].values[0]
    else:
        print("Warning: station {} has no co-ordinates!".format(data.station[S]))
data["station_bottleid"] = [
    row.station + "_" + str(row.bottleid) for _, row in data.iterrows()
]

#%% Convert units from mg/l to μmol/l
data["doc_vol"] = 1e3 * data.doc_gvol / ks.molar.mass["C"]
data["poc_vol"] = 1e3 * data.poc_gvol / ks.molar.mass["C"]
data["nitrate_vol"] = 1e3 * data.nitrate_gvol / ks.molar.mass["N"]
data["nitrite_vol"] = 1e3 * data.nitrite_gvol / ks.molar.mass["N"]
data["dn_vol"] = 1e3 * data.dn_gvol / ks.molar.mass["N"]
data["ammonia_vol"] = 1e3 * data.ammonia_gvol / ks.molar.mass["N"]
data["din_vol"] = data.nitrate_vol + data.nitrite_vol + data.ammonia_vol
data["silicate_vol"] = 1e3 * data.silicate_gvol / ks.molar.mass["Si"]
data["phosphate_vol"] = (
    data.phosphate_gvol / ks.molar.mass["P"]
)  # this one provided in μg/l, not mg/l
data["dp_vol"] = 1e3 * data.dp_gvol / ks.molar.mass["P"]
data["sulfate_vol"] = 1e3 * data.sulfate_gvol / ks.molar.mass["SO4"]
data["chloride_vol"] = 1e3 * data.chloride_gvol / ks.molar.mass["Cl"]

# Set negative nutrients to zero
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
    nvol = nutrient + "_vol"
    data.loc[data[nvol] < 0, nvol] = 0

# Assign nutrient uncertainties from RWS
data["silicate_vol_unc"] = data.silicate_vol * 0.15
data["phosphate_vol_unc"] = data.phosphate_vol * 0.25
data["nitrate_vol_unc"] = data.nitrate_vol * 0.3
data["nitrite_vol_unc"] = data.nitrite_vol * 0.3
data["ammonia_vol_unc"] = data.ammonia_vol * 0.15
data["doc_vol_unc"] = data.doc_vol * 0.2

#%% Import lab sheet and copy across salinities (for up to v6 only)
# lab = pd.read_excel(
#     "data/co2data-20200406.xlsx",
#     sheet_name="2018_2019 All Final Data",
#     header=2,
#     na_values=["#N/B", -999],
# )
# data["salinity_nioz"] = np.nan
# data["datfile_stem"] = ""
# for i in data.index:
#     il = (lab.station == data.loc[i].station) & (lab.bottleid == data.loc[i].bottleid)
#     if sum(il) == 1:
#         data.loc[i, "salinity_nioz"] = lab.salinity_nioz[il].values
#         data.loc[i, "datfile_stem"] = lab.station_bottleid[il].values
data["salinity"] = data.salinity_rws
# for i in data.index:
#     if np.isnan(data.loc[i, "salinity"]):
#         data.loc[i, "salinity"] = data.loc[i].salinity_nioz

#%% Get unique stations table and assign properties
stations = pd.DataFrame(index=np.unique(data.station))
stations["latitude"] = np.nan
stations["longitude"] = np.nan
for station in stations.index:
    SL = data.station == station
    stations.loc[station, "latitude"] = np.unique(data.latitude[SL])
    stations.loc[station, "longitude"] = np.unique(data.longitude[SL])
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

# Get station descriptions
stations["description"] = [
    allstations.loc[allstations["U meetpunt"] == station, "U omschr meetpunt"].values[0]
    for station in stations.index
]

#%% Add more sampling date variants
data["datetime"] = pd.to_datetime(data.datetime, format="%d-%m-%Y %H:%M")
data["datenum"] = mdates.date2num(data.datetime)
data["day_of_year"] = data.datetime.dt.dayofyear

# Add depth info
data["depth"] = -data.height_cm / 100

# Save for next step of Python analysis
with open("pickles/data0_stations_groups_v10.pkl", "wb") as f:
    pickle.dump((data, stations, groups), f)
