import pathlib
import pandas as pd

# this script builds a .txt file for each metric that contains the percentages of non-empty fields for each station
# for each year
# I used it to decide which years and stations to use for the final analysis; generally, I tried to pick sequences
# of consecutive years with a score of at least 0.95
# in practice, this usually meant discarding everything before 2014-2017 or so

# this script should be run after pollution_parsing.py
# note that this script was only used for informative purposes and is not necessary to build visualisations, compile
# report or run experiments; the knowledge gained from this script is already implemented in pollution_final_transform.py

metrics_of_interest = ["CO_1g", "NO2_1g", "O3_1g", "SO2_1g", "PM10_1g", "PM25_1g"]
dir_path = "preprocessed_pollution_data"
dir_path = pathlib.Path(dir_path)

# for each metric
for metric in metrics_of_interest:
    # build a dictionary mapping each station to a list of tuples, each tuple containing a year and the
    # share of non-missing values in that year
    results = {}
    path = dir_path / metric
    for file_path in path.iterdir():
        if file_path.name.endswith(".csv"):
            df = pd.read_csv(file_path, index_col=0)
            for station_code in df.columns:
                if not station_code in results.keys():
                    results[station_code] = []
                results[station_code].append(
                    (file_path.name, 1 - df.isnull().sum()[station_code] / len(df))
                )
    # write report
    with open(path / "values.txt", "w+") as output_file:
        for station_code, year_ratio_pair_list in results.items():
            output_file.write(f"{station_code}:\n")
            # sort by years
            year_ratio_pair_list.sort(key=lambda x: int(x[0][0:3]))
            for year, ratio in year_ratio_pair_list:
                output_file.write(f"{year} : {ratio:.2f}\n")
            output_file.write("\n")