# this script performs the selection and some final cleaning of pollution data
# it should be run after pollution_parsing
# it is performed seperately because it relies on knowledge gleaned from the
# previous step (preliminary_analysis)

import pandas as pd
import pathlib

import data_loading

metrics = ["CO_1g", "NO2_1g", "O3_1g", "SO2_1g", "PM10_1g", "PM25_1g"]
final_year = 2022


# build final pollution data .csv files
# format is a single file for each metric
# station - name of the station to source data from
# starting_years - a list of the starting year for each metric, or 0 if the metric should be omitted
# destination_path - path to the destination directory
def build_final_data_file(
    station, starting_years, destination_path=pathlib.Path("final_pollution_data")
):
    origin_path = pathlib.Path("preprocessed_pollution_data")
    # for station,starting_years in station_starting_years:
    for metric, start_year in zip(metrics, starting_years):
        if start_year > 0:
            # load data
            data = data_loading.load_and_combine_year_csvs(
                origin_path / metric, start_year, final_year
            )
            # pick the right station
            data = data[[station]]
            # change column name from station code to metric name
            data.rename(columns={station: metric[:-3]}, inplace=True)
            # fill in missing values
            data.interpolate(inplace=True, limit_direction="both")
            # save to csv
            destination_path.mkdir(parents=True, exist_ok=True)
            data.to_csv(destination_path / (metric + ".csv"))


if __name__ == "__main__":
    build_final_data_file("DsWrocWybCon", [2015, 2014, 2015, 2015, 2016, 2017])
    # an alternate dataset, currently used only for the wind direction comparison
    build_final_data_file("DsWrocAlWisn", [2015, 2013, 0, 0, 0, 2017], pathlib.Path("final_pollution_data_alt"))
