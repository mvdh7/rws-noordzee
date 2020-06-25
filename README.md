# Rijkswaterstaat North Sea monitoring programme

## Requirements

A Conda environment that can be used to run the analysis scripts is specified in `environment.yml`.  This will be updated with each new release.

## Versioning

Analysis is conducted periodically in batches, where each batch consists of the samples from a few months of observations.  The major version number refers to how many batches of analysis are included up to this point.  The minor version number refers to updated dataset releases where the values have changed.

### Currently released versions

  * **v6.1** (25 June 2020).

## Datasets

All in subdirectory [data](data):

  * `Coordinaten_verzuring_20190429.xlsx` received from Sharyn Ossebaar 2019-05-09. Contains locations of each sampling location. Have added extra columns with latitude/longitude values received from Karel Bakker on 2019-05-20.
  * Lab measurement datasets (TA, DIC and pH):
    * `archive\co2data-20190429.xlsx` received from Sharyn Ossebaar 2019-04-29 as `2018-All Final Data.xlsx`
    * `archive\co2data-20190520.xlsx` received from Karel Bakker 2019-05-20 as `All Final Data.xlsx`. Added header row to sheet *SUMMARY All Final DATA*, for easier Pandas import.
    * `co2data-20200406.xlsx` received from Sharyn Ossebaar 2020-04-06 as `All Final Data.xlsx`. Added header row to sheet *2018_2019 All Final Data*, for easier Pandas import.
  * `Bottlefile_NIOZ_2019_20190412.xlsx` received from Sharyn Ossebaar 2019-05-22. Contains bottle file metadata for the samples including nutrients and in situ temperature. Added extra header row for easier importing.
  * `Bottle Files/*` copied from Zeus 2020-04-06.
  * `pH Spec/*` copied from Zeus 2020-04-06.  Some files modified and those have been renamed with suffix `_MPH`.
  * `VINDTA#15 Data/*` copied from Zeus 2020-04-06.

## Data import and processing

### Workflow

All in Python:

  1. **Run `data0.py`.**  This imports the raw data from all various sources and tidies it up ready for processing.  Generates `pickles/data0-XXXXXX.pkl`, where `XXXXXX` corresponds to the `dataversion` as in the lab measurement dataset file.
  2. **Run `data1.py`.**  This imports the output of `data0.py` and performs data calibration and processing.  Generates `pickles/data1-XXXXXX.pkl` for making figures, and Excel spreadsheets of the results where `results/RWS_Noordzee_v6_raw.xlsx` is used for final reporting.

### More details

The current version of `XXXXXX` is `dataloading = 20200406`.

`data0.py`: import raw carbonate system data from the NIOZ lab techs and nutrient data from RWS. Merge together along with sampling dates and times. Determine different stations and groups of stations and assign colours for plotting.

  * Import station location data from `data/Coordinaten_verzuring_20190429.xlsx`.
  * Add to measurements from `data/co2data-XXXXXX.xlsx`.
  * Create merged file `pickles/co2data-XXXXXX.csv` for import into MATLAB.
  * Create `stations` and `groups` pivot tables and assign station colours and groups.
  * Pickle `data`, `stations` and `groups` in `pickles/data0-XXXXXX.pkl` ready for `data1.py`.

`data1.py`: recalibrate the carbonate system data measured at NIOZ.
