#initial processing of pollution data
import numpy as np
import pandas as pd
import pathlib

#codes of stations in Wrocław; we can discard columns from all other stations
station_codes = [
    "DsWrocAlWisn",
    "DsWrocBartni",
    "DsWrocGrun",
    "DsWrocKorzPMArch",
    "DsWrocKrom",
    "DsWrocNaGrob",
    "DsWrocOlsz",
    "DsWrocOpor",
    "DsWrocOrzech",
    "DsWrocOrzechArch",
    "DsWrocPret",
    "DsWrocSklad",
    "DsWrocTech",
    "DsWrocUkr",
    "DsWrocWie",
    "DsWrocWybCon",
]

#some stations had changed their codes at some point, and thus figure under different codes in different spreadsheets
#we need to standarize them to properly identify stations
code_updates = {
    "DsWrocBartA":"DsWrocBartni",
    "DsWrocWisA":"DsWrocAlWisn",
    "DsWrocGrobla":"DsWrocNaGrob",
    "DsWrocKorzA":"DsWrocWybCon"
}

metrics_of_interest = ["CO_1g","NO2_1g","O3_1g","SO2_1g","PM10_1g","PM25_1g"]

MIN_YEAR = 2000
MAX_YEAR = 2023

#use the station codes as column names
def fix_header(data): 
    if data.index[0] == "Kod stacji" or data.index.isnull()[0]:
        data.columns = data.iloc[0]
        data = data[1:]
    return data

#transform the data from a .xlsx file to an usable dataframe
def process_spreadsheet(path):
    #index_col=0 ensures that the first column (timestamps) will be used as row indices (otherwise artificial 
    #numerical indices would be added)
    data = pd.read_excel(path,index_col=0)
    #make the station codes column names
    data = fix_header(data)
    #update old station codes to present ones
    data.rename(axis="columns",mapper=code_updates,inplace=True)
    #remove data from outside Wroclaw
    data.drop([col for col in data.columns if not col in station_codes],axis=1,inplace=True)
    #some of the files have up to five unnecessary header rows which we need to get rid of
    #they may or may not be named, hence we also drop rows with null indices
    data = data[data.index.notnull()]
    data.drop("Wskaźnik",inplace=True,errors="ignore")
    data.drop("Jednostka",inplace=True,errors="ignore")
    data.drop("Czas uśredniania",inplace=True,errors="ignore")
    data.drop("Czas pomiaru",inplace=True,errors="ignore")
    data.drop("Kod stanowiska",inplace=True,errors="ignore")
    #check if the cells contain commas, if so, they need to be replaced with dots and the values cast to float
    if not data.empty and (',' in str(data.loc[data[data.columns[0]].notnull()][data.columns[0]].iloc[10])):
        data = data.apply(lambda x: x.str.replace(',', '.').astype(float), axis=1)
    #there is at least one pollution data file where each subsequent measurement is shifted forward by a few miliseconds
    #relative to previous one, this accumulates to about 40 seconds over a year
    #we need to handle it or else dataframe joins won't work properly
    #the following line looks a bit confusing, but all it does is convert the indices to datetime objects and round them
    #down to full hour
    data.index = pd.Series(pd.to_datetime(data.index,format='mixed')).dt.floor(freq="h")
    return data

#build .csv files containing only the data we are interested in
#the files are split by metric and then by year
def extract_relevant_data():
    origin_path = "raw_pollution_data"
    origin_path = pathlib.Path(origin_path)
    destination_path = "preprocessed_pollution_data"
    destination_path = pathlib.Path(destination_path)
    years = range(MIN_YEAR,MAX_YEAR)
    for metric in metrics_of_interest:
        for year in years:
            file_path = ((origin_path / str(year)) / (str(year) + "_" +  metric + ".xlsx"))
            if file_path.exists():
                df = process_spreadsheet(file_path)
                if not df.empty:
                    (destination_path / metric).mkdir(parents=True, exist_ok=True)
                    df.to_csv(str((destination_path / metric / str(year)).resolve()) + ".csv")

if __name__ == "__main__":
    extract_relevant_data()

