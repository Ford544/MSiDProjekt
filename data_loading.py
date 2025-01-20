import pathlib
import pandas as pd

# default year range
# excludes final year, as many experiments set it aside as testing set
STARTYEAR = 2015
ENDYEAR = 2021


# helper function that reads csv files named like [year].csv in a given directory and concatenates their contents
# into a single dataframe
def load_and_combine_year_csvs(path, startyear, endyear):
    path = pathlib.Path(path)
    data = pd.DataFrame()
    for year in range(startyear, endyear + 1):
        df = pd.read_csv(path / (str(year) + ".csv"), index_col=0)
        if data.empty:
            data = df
        else:
            data = pd.concat([data, df], axis=0)
    return data


# build a dataframe containing all hourly weather data
def load_hourly_weather_data():
    weather = load_and_combine_year_csvs(
        "preprocessed_weather_data\\hourly", 2013, 2022
    )
    weather.index = pd.to_datetime(weather.index)
    return weather


# build a dataframe containing hourly pollution data in given year range and for given metrics
# returns all data up to 2021 by default
def load_pollution_data(
    startyear=STARTYEAR,
    endyear=ENDYEAR,
    metrics=["CO_1g", "NO2_1g", "O3_1g", "SO2_1g", "PM10_1g", "PM25_1g"],
    path="final_pollution_data",
):
    data = pd.DataFrame()
    for metric in metrics:
        if pathlib.Path(path + f"\\{metric}.csv").exists():
            if data.empty:
                data = pd.read_csv(path + f"\\{metric}.csv", index_col=0)
            else:
                data = data.join(
                    pd.read_csv(path + f"\\{metric}.csv", index_col=0), how="inner"
                )
    data.index = pd.to_datetime(data.index)
    return data.loc[f"{startyear}-1-1":f"{endyear}-12-31"]


# build a dataframe containing hourly pollution and weather data
def get_full_hourly_dataset(
    startyear=STARTYEAR,
    endyear=ENDYEAR,
    metrics=["CO_1g", "NO2_1g", "O3_1g", "SO2_1g", "PM10_1g", "PM25_1g"],
    path="final_pollution_data",
):
    weather = load_hourly_weather_data()
    pollution = load_pollution_data(startyear, endyear, metrics, path)
    return weather.join(pollution, how="inner")


# transforms the WindDir variable from numerical to categorical
def categorize_wind_dir(df):
    df = df.copy()
    wind_dir_cats = pd.Categorical(
        ["NULL", "N", "NE", "E", "SE", "S", "SW", "W", "NW"], ordered=True
    )

    # pick wind direction category based on the angle in degrees
    def getWindDirCat(n):
        if n == 0:
            return wind_dir_cats[0]
        if 0 < n <= 22.5 or 337.5 < n <= 360:
            return wind_dir_cats[1]
        i = 22.5
        for cat in wind_dir_cats[2:]:
            if i < n <= i + 45:
                return cat
            i += 45

    df.WindDir = df.WindDir.apply(getWindDirCat)
    return df


# encodes a categorical variable as a set of binary columns
def make_dummy(df, column):
    cat = pd.get_dummies(df[column], dtype=int)
    df.drop(column, inplace=True, axis=1)
    df = pd.concat([df, cat], axis=1)
    return df
