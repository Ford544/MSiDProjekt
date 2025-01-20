#this scripts build all visualisations used in the report
#it should be run after pollution_parsing, pollution_final_transform and weather_parsing_full

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pathlib
from sklearn.linear_model import LinearRegression

from data_loading import get_full_hourly_dataset, categorize_wind_dir

img_dir_path = pathlib.Path("visualisations")
metrics = ["CO","NO2","O3","SO2","PM10","PM25"]
#WindDir is skipped because it's not exactly numerical
#Precip is skipped because it requires additional preprocessing
hourly_weather_variables = ["WindSpeed","Temp","Humidity","Pressure"]

#make a plot of a relationship between two columns in a dataframe
#takes a dateframe, two column names, and a path to the destination directory
#saves a scatterplot with a trend line
def save_scatter_plot_with_linear_regression(data,x,y,destination):
    model_lin = LinearRegression()
    X = data[x].values.reshape(-1,1)
    Y = data[y].values.reshape(-1,1)
    model_lin.fit(X, Y)
    X_test = np.linspace(start=data[x].values.min(), stop=data[x].values.max(), num=300)
    Y_pred = model_lin.predict(X_test.reshape(-1,1))
    sns.scatterplot(data=data,x=x,y=y, s=8)
    plt.plot(X_test, Y_pred, color='tab:orange', linewidth=3)
    plt.savefig(destination / f"{x}x{y}_scatter.png")
    plt.clf()

#create histograms of the variables and metrics
def make_histograms(fhd):
    destination = img_dir_path / "histograms"
    (destination).mkdir(parents=True, exist_ok=True)
    for col in fhd.columns:
        sns.histplot(data=fhd,x=col)
        plt.savefig(destination / (col + "_hist.png"))
        plt.clf()
    #precipitation is handled seperately, as we only want to include those rows
    #where precipitation was measured (hour is divisible by 6)
    df = fhd[fhd.index.hour % 6 == 0]
    sns.histplot(data=df,x="Precip")
    plt.savefig(destination / ("Precip" + "_hist.png"))
    plt.clf()

#create correlation heatmaps
def make_corr_heatmaps(fhd):
    destination = img_dir_path / "corr_heatmaps"
    (destination).mkdir(parents=True, exist_ok=True)
    #correlation between weather variables
    #rolling window is used to account for sparse precipitation measurements
    sns.heatmap(fhd[hourly_weather_variables + ["Precip"]].rolling(window=6,min_periods=1).mean().corr(), cmap="YlGnBu", annot=True) 
    plt.savefig(destination / "weather_corr_rolling_heatmap.png")
    plt.clf()
    #correlation between pollution metrics
    sns.heatmap(fhd[metrics].corr(), cmap="YlGnBu", annot=True) 
    plt.savefig(destination / "pollution_corr_heatmap.png")
    plt.clf()
    #correlation between weather variables and pollution metrics
    #rolling window is used to account for sparse precipitation measurements
    sns.heatmap(fhd.rolling(window=6,min_periods=1).mean().corr()[hourly_weather_variables+["Precip"]].loc[metrics], cmap="YlGnBu", annot=True) 
    plt.savefig(destination / "pollutionxweather_corr_heatmap.png")
    plt.clf()

#make scatterplots with trend line of all variable-metric pairs (except for WindDir)
def make_metric_variable_plots(fhd):
    destination = img_dir_path / "corr_plots"
    (destination).mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        for var in hourly_weather_variables:
            save_scatter_plot_with_linear_regression(fhd,var,metric,destination)
    #precipitation is handled seperately, as we only want to include those rows
    #where precipitation was measured (hour is divisible by 6)
    fhd_precip = fhd[fhd.index.hour % 6 == 0]
    for metric in metrics:
        save_scatter_plot_with_linear_regression(fhd_precip,"Precip",metric,destination)

#make scatterplots with trend line of all variable-variable pairs (except for WindDir)
def make_variable_variable_plots(fhd):
    destination = img_dir_path / "weather_corr_plots"
    (destination).mkdir(parents=True, exist_ok=True)
    for i,var1 in enumerate(hourly_weather_variables):
        for var2 in hourly_weather_variables[i+1:]:
            save_scatter_plot_with_linear_regression(fhd,var2,var1,destination)
    #precipitation is handled seperately, as we only want to include those rows
    #where precipitation was measured (hour is divisible by 6)
    fhd_precip = fhd[fhd.index.hour % 6 == 0]
    for var in hourly_weather_variables:
        save_scatter_plot_with_linear_regression(fhd_precip,"Precip",var,destination)

#make plots of time (daily, weekly and yearly) of pollution metrics and chosen weather variables
def make_time_cycle_plots(fhd):
    destination = img_dir_path / "cycles"
    (destination).mkdir(parents=True, exist_ok=True)
    hours_data = fhd.groupby(lambda x : x.hour).mean()
    days_of_week_data = fhd.groupby(lambda x : x.weekday()).mean()
    days_of_week_data.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    months_data = fhd.groupby(lambda x : x.month).mean() 
    months_data.index = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    for metric in metrics:
        #daily cycle
        sns.lineplot(x=hours_data.index, y=hours_data[metric],marker='o').set_xlabel("Hour")
        plt.savefig(destination / f"hourly_{metric}.png")
        plt.clf()
        #daily cycle with temperature
        sns.lineplot(x=hours_data.index, y=hours_data[metric],marker='o').set_xlabel("Hour")
        ax2 = plt.twinx()
        sns.lineplot(data=hours_data.Temp, color="r", ax=ax2)
        plt.savefig(destination / f"hourly_{metric}_temp.png")
        plt.clf()
        #weekly cycle
        sns.lineplot(x=days_of_week_data.index, y=days_of_week_data[metric],marker='o').set_xlabel("Day of week")
        plt.savefig(destination / f"weekly_{metric}.png")
        plt.clf()
        #yearly cycle
        sns.lineplot(x=months_data.index, y=months_data[metric],marker='o').set_xlabel("Month")
        plt.savefig(destination / f"monthly_{metric}.png")
        plt.clf()
    #chosen weather cycles
    destination = img_dir_path / "weather_cycles"
    (destination).mkdir(parents=True, exist_ok=True)
    sns.lineplot(x=hours_data.index, y=hours_data["WindSpeed"],marker='o').set_xlabel("Hour")
    plt.savefig(destination / f"hourly_wind_speed.png")
    plt.clf()

#make boxplots of pollution metrics grouped by wind direction
def make_wind_dir_plots(fhd):
    wind_dir_dataset = categorize_wind_dir(fhd)
    destination = img_dir_path / "wind_dir_pollution_patterns"
    (destination).mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        sns.boxplot(data=wind_dir_dataset,x=metric,y="WindDir",showfliers=False,order=["NULL","N","NE","E","SE","S","SW","W","NW"])
        plt.savefig(destination / f"wind_dir_{metric}.png")
        plt.clf()

#make boxplots of pollution metrics grouped by wind direction for the alternate dataset
def make_alt_wind_dir_plots():
    alt = get_full_hourly_dataset(path = "final_pollution_data_alt")
    wind_dir_dataset = categorize_wind_dir(alt)
    destination = img_dir_path / "wind_dir_pollution_alt_patterns"
    (destination).mkdir(parents=True, exist_ok=True)
    for metric in ["CO","NO2","PM25"]:
        sns.boxplot(data=wind_dir_dataset,x=metric,y="WindDir",showfliers=False,order=["NULL","N","NE","E","SE","S","SW","W","NW"])
        plt.savefig(destination / f"wind_dir_{metric}.png")
        plt.clf()

if __name__ == "__main__":
    fhd = get_full_hourly_dataset()

    #HISTOGRAMS
    make_histograms(fhd)

    #CORRELATION MATRICES
    make_corr_heatmaps(fhd)

    #METRIC : VARIABLE PAIRS
    make_metric_variable_plots(fhd)

    #VARIABLE : VARIABLE PAIRS
    make_variable_variable_plots(fhd)

    #CYCLES
    make_time_cycle_plots(fhd)

    #WIND DIR PATTERNS
    make_wind_dir_plots(fhd)
    
    #WIND DIR PATTERNS - ALT DATASET
    make_alt_wind_dir_plots()
