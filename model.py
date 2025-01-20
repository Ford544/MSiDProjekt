from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from data_loading import get_full_hourly_dataset, categorize_wind_dir, make_dummy

lastyear = 2022
metrics = ["CO", "NO2", "O3", "SO2", "PM10", "PM25"]
hourly_weather_variables = [
    "WindDir",
    "WindSpeed",
    "Temp",
    "Humidity",
    "Pressure",
    "Precip",
]
# default target
target = "CO"

months = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
hours = list(map(lambda x: "H" + str(x), range(0, 24)))


# create and encode categorical variables from hour, weekday and month
def make_time_variables(df):
    df["Month"] = df.index.map(lambda x: months[x.month - 1])
    df = make_dummy(df, "Month")
    df["Hour"] = df.index.map(lambda x: "H" + str(x.hour))
    df = make_dummy(df, "Hour")
    df["Weekday"] = df.index.map(lambda x: weekdays[x.weekday() - 1])
    df = make_dummy(df, "Weekday")
    return df


# Prepare the dataset for modeling
# target - the name of the target metric
# make_time_vars - if True, create and encode categorical variables for hour, weekday and month
# select_vars - if not None, only those variables will be returned
# returns two dataframes, one for predictor variables and one for target
def prepare_dataset(target, make_time_vars=False, select_vars=None):
    # Load dataset
    df = get_full_hourly_dataset(endyear=lastyear)
    # Pick only variables for X (discard metrics)
    X = df[hourly_weather_variables]
    # Average precipitation
    a = X["Precip"].rolling(window=6, min_periods=1, center=True).mean()
    X.loc[X.index, "Precip"] = a
    # Categorize WindDir
    X = make_dummy(categorize_wind_dir(X), "WindDir")
    # Standarize variable values
    a = X[["WindSpeed", "Temp", "Humidity", "Pressure", "Precip"]]
    X[["WindSpeed", "Temp", "Humidity", "Pressure", "Precip"]] = X[
        ["WindSpeed", "Temp", "Humidity", "Pressure", "Precip"]
    ].astype(float)
    X.loc[X.index, ["WindSpeed", "Temp", "Humidity", "Pressure", "Precip"]] = (
        a - a.mean()
    ) / a.std()
    # Pick a chosen metric for Y
    Y = df[[target]]
    # Categorize time
    if make_time_vars:
        X = make_time_variables(X)
    # Select variables
    if select_vars is not None:
        X = X[select_vars]
    return X, Y


# make a time-based training-and-test dataset split
# specifically, it uses the last year as test and all previous years as training
def get_dataset_split_by_time(target, make_time_vars=False, select_vars=None):
    X, Y = prepare_dataset(target, make_time_vars, select_vars)
    X_train = X[X.index.year < lastyear].values
    X_test = X[X.index.year == lastyear].values
    Y_train = Y[Y.index.year < lastyear].values.flatten()
    Y_test = Y[Y.index.year == lastyear].values.flatten()
    return X_train, Y_train, X_test, Y_test


# linear regression, with random and time-based splits, with and without time data
def experiment1():
    print(
        "EXP1: linear regression, with random and time-based splits, with and without time data"
    )
    # time-based split
    for use_time in [True, False]:
        X_train, Y_train, X_test, Y_test = get_dataset_split_by_time(target, use_time)
        for degree in [1, 2, 3]:
            # we exclude this test because it requires too much resources (memory)
            if use_time and degree == 3:
                continue
            if degree == 1:
                model_lin = LinearRegression()
                model_lin.fit(X_train, Y_train)
                RMSE = root_mean_squared_error(Y_test, model_lin.predict(X_test))
                print(f"Time based use_time={use_time} degree={degree} RMSE={RMSE}")
            else:
                model_lin = LinearRegression()
                gen_features = PolynomialFeatures(
                    degree=degree, include_bias=True, interaction_only=False
                )
                model_lin.fit(gen_features.fit_transform(X_train), Y_train)
                RMSE = root_mean_squared_error(
                    Y_test, model_lin.predict(gen_features.fit_transform(X_test))
                )
                print(
                    f"Linear time based use_time={use_time} degree={degree} RMSE={RMSE}"
                )
    # random split
    for use_time in [True, False]:
        X, Y = prepare_dataset(target, use_time)
        for degree in [1, 2, 3]:
            # we exclude this test because it requires too much resources (memory)
            if use_time and degree == 3:
                continue
            count = 0
            for i in range(0, 5):
                X_train, X_test, Y_train, Y_test = train_test_split(
                    X, Y, train_size=0.83
                )
                if degree == 1:
                    model_lin = LinearRegression()
                    model_lin.fit(X_train, Y_train)
                    RMSE = root_mean_squared_error(Y_test, model_lin.predict(X_test))
                else:
                    model_lin = LinearRegression()
                    gen_features = PolynomialFeatures(
                        degree=degree, include_bias=True, interaction_only=False
                    )
                    model_lin.fit(gen_features.fit_transform(X_train), Y_train)
                    RMSE = root_mean_squared_error(
                        Y_test, model_lin.predict(gen_features.fit_transform(X_test))
                    )
                count += RMSE
            print(f"Linear use_time={use_time} degree={degree} RMSE={count/5}")


# tree and forest, with random and time-based splits, with and without time data
def experiment2():
    print(
        "EXP2: tree and forest, with random and time-based splits, with and without time data"
    )
    for depth in range(3, 10):
        for use_time in [True, False]:
            # time-based split
            X_train, Y_train, X_test, Y_test = get_dataset_split_by_time(
                target, use_time
            )
            model_tree = tree.DecisionTreeRegressor(max_depth=depth)
            model_tree.fit(X_train, Y_train)
            RMSE_TRE = root_mean_squared_error(Y_test, model_tree.predict(X_test))
            print(
                f"Decision tree, time based, max_depth={depth}, use_time={use_time} RMSE={RMSE_TRE:0.3}"
            )
            # random split
            X, Y = prepare_dataset(target, use_time)
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, train_size=0.83, random_state=42
            )
            Y_train = Y_train.values.flatten()
            Y_test = Y_test.values.flatten()
            model_tree = tree.DecisionTreeRegressor(max_depth=depth)
            model_tree.fit(X_train, Y_train)
            RMSE_TRE = root_mean_squared_error(Y_test, model_tree.predict(X_test))
            print(
                f"Decision tree, random, max_depth={depth}, use_time={use_time} RMSE={RMSE_TRE:0.3}"
            )

    for depth in range(3, 10):
        for use_time in [True, False]:
            # time-based split
            X_train, Y_train, X_test, Y_test = get_dataset_split_by_time(
                target, use_time
            )
            model_tree = RandomForestRegressor(max_depth=depth)
            model_tree.fit(X_train, Y_train)
            RMSE_TRE = root_mean_squared_error(Y_test, model_tree.predict(X_test))
            print(
                f"Random forest, time based, max_depth={depth}, use_time={use_time} RMSE={RMSE_TRE:0.3}"
            )
            # random split
            X, Y = prepare_dataset(target, use_time)
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, train_size=0.83, random_state=42
            )
            Y_train = Y_train.values.flatten()
            Y_test = Y_test.values.flatten()
            model_tree = RandomForestRegressor(max_depth=depth)
            model_tree.fit(X_train, Y_train)
            RMSE_TRE = root_mean_squared_error(Y_test, model_tree.predict(X_test))
            print(
                f"Random forest, random, max_depth={depth}, use_time={use_time} RMSE={RMSE_TRE:0.3}"
            )


# model comparison for all metrics, with time based split and time data
def experiment3():
    print("EXP3: model comparison for all metrics, with time based split and time data")
    for target in metrics:
        X_train, Y_train, X_test, Y_test = get_dataset_split_by_time(target, True)
        model_lin = LinearRegression()
        model_lin.fit(X_train, Y_train)
        RMSE = root_mean_squared_error(Y_test, model_lin.predict(X_test))
        print(f"{target} prediction: Time-only linear-1 RMSE={RMSE}")
        print(f"R^2 score: {r2_score(Y_test,model_lin.predict(X_test))}")

        model_lin = LinearRegression()
        gen_features = PolynomialFeatures(
            degree=2, include_bias=True, interaction_only=False
        )
        model_lin.fit(gen_features.fit_transform(X_train), Y_train)
        RMSE = root_mean_squared_error(
            Y_test, model_lin.predict(gen_features.fit_transform(X_test))
        )
        print(f"{target} prediction: Time-only linear-2 RMSE={RMSE}")
        print(
            f"R^2 score: {r2_score(Y_test,model_lin.predict(gen_features.fit_transform(X_test)))}"
        )

        for depth in [4, 7, 16, 24, 43]:
            model_tree = RandomForestRegressor(max_depth=depth)
            model_tree.fit(X_train, Y_train)
            RMSE_TRE = root_mean_squared_error(Y_test, model_tree.predict(X_test))
            print(f"{target} prediction: Time only forest-{depth}: {RMSE_TRE:0.3}")
            print(f"R^2 score: {r2_score(Y_test,model_tree.predict(X_test))}")


# model comparison, time data only
def experiment4():
    print("EXP4: model comparison, time data only")
    for target in metrics:
        X_train, Y_train, X_test, Y_test = get_dataset_split_by_time(
            target, True, months + weekdays + hours
        )
        model_lin = LinearRegression()
        model_lin.fit(X_train, Y_train)
        RMSE = root_mean_squared_error(Y_test, model_lin.predict(X_test))
        print(f"{target} prediction: Time-only linear-1 RMSE={RMSE}")
        print(f"R^2 score: {r2_score(Y_test,model_lin.predict(X_test))}")

        model_lin = LinearRegression()
        gen_features = PolynomialFeatures(
            degree=2, include_bias=True, interaction_only=False
        )
        model_lin.fit(gen_features.fit_transform(X_train), Y_train)
        RMSE = root_mean_squared_error(
            Y_test, model_lin.predict(gen_features.fit_transform(X_test))
        )
        print(f"{target} prediction: Time-only linear-2 RMSE={RMSE}")
        print(
            f"R^2 score: {r2_score(Y_test,model_lin.predict(gen_features.fit_transform(X_test)))}"
        )

        for depth in [7, 16, 24, 43]:
            model_tree = RandomForestRegressor(max_depth=depth)
            model_tree.fit(X_train, Y_train)
            RMSE_TRE = root_mean_squared_error(Y_test, model_tree.predict(X_test))
            print(f"{target} prediction: Time only forest-{depth}: {RMSE_TRE:0.3}")
            print(f"R^2 score: {r2_score(Y_test,model_tree.predict(X_test))}")


# linear regression, non-categorical data only
def experiment5():
    print("EXP5: linear regression, non-categorical data only")
    for target in metrics:
        for degree in range(2, 6):
            X_train, Y_train, X_test, Y_test = get_dataset_split_by_time(
                target, False, ["WindSpeed", "Temp", "Humidity", "Pressure", "Precip"]
            )
            model_lin = LinearRegression()
            gen_features = PolynomialFeatures(
                degree=degree, include_bias=True, interaction_only=False
            )
            model_lin.fit(gen_features.fit_transform(X_train), Y_train)
            RMSE = root_mean_squared_error(
                Y_test, model_lin.predict(gen_features.fit_transform(X_test))
            )
            print(
                f"{target} prediction: non-categorical-only linear-{degree} RMSE={RMSE}"
            )


if __name__ == "__main__":
    experiment1()
    experiment2()
    experiment3()
    experiment4()
    experiment5()
