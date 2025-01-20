#processing of the weather data

import pandas as pd
import pathlib

MIN_YEAR = 2013
MAX_YEAR = 2023


# this function takes in a path to a file, a list containing indices of columns which should be included in the output,
# and a list of names to be assigned for those columns
# returns a formatted dataframe
def read_meteorology_file(path, indices_of_interest, column_names):
    # NOTE I had to change the encoding because using default utf-8 produced an error
    # one side effect of this is that Ł in WROCŁAW gets turned into £
    # I'm going to drop that column anyway, so it shouldn't be a problem
    df = pd.read_csv(path, header=None, encoding="unicode_escape")
    # extract the columns we care about and discard the rest
    datetime_indices = [2, 3, 4, 5]
    datetime_column_names = ["Year", "Month", "Day", "Hour"]
    df = df[datetime_indices + indices_of_interest]
    df.columns = datetime_column_names + column_names
    # now we combine the datatime fields into singular datetime objects which we then use as row indices
    df = merge_date_hourly(df)
    # this is just in case there are some missing values
    # It doesn't appear that there are any, though; the meteorology folks treat their job seriously
    df.interpolate(inplace=True)
    return df


# take in a dataframe and transform datetime columns into row indices
def merge_date_hourly(df):
    df.index = pd.to_datetime(
        df.Year.astype(str)
        + "/"
        + df.Month.astype(str)
        + "/"
        + df.Day.astype(str)
        + "/"
        + df.Hour.astype(str),
        format=r"%Y/%m/%d/%H",
    )
    df.drop("Year", axis="columns", inplace=True)
    df.drop("Month", axis="columns", inplace=True)
    df.drop("Day", axis="columns", inplace=True)
    df.drop("Hour", axis="columns", inplace=True)
    return df


# transform raw data into directly usable csv files
def extract_relevant_data():
    origin_path = "raw_weather_data"
    origin_path = pathlib.Path(origin_path)
    destination_path = "preprocessed_weather_data"
    destination_path = pathlib.Path(destination_path)
    # year boundaries chosen manually based on the output of preliminary_analysis.py
    years = range(MIN_YEAR, MAX_YEAR)
    # HOURLY DATA
    for year in years:
        dir_path = origin_path / "hourly"
        # 23,25,29,37,41,48 are respective indices of the chosen variables in the raw dataset
        data = read_meteorology_file(
            dir_path / ("s_t_424_" + str(year) + ".csv"),
            [23, 25, 29, 37, 41, 48],
            ["WindDir", "WindSpeed", "Temp", "Humidity", "Pressure", "Precip"],
        )

        (destination_path / "hourly").mkdir(parents=True, exist_ok=True)
        data.to_csv(destination_path / "hourly" / (str(year) + ".csv"))


if __name__ == "__main__":
    extract_relevant_data()
