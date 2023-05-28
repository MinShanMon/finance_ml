from flask import Flask, jsonify, request
import pymysql
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.metrics import mean_squared_error
import itertools
import pandas as pd
import numpy as np
import pymysql
import warnings
from flask import Flask, request, jsonify
warnings.filterwarnings("ignore") 
app = Flask(__name__)
# load the data,and save the related column
# df = pd.read_csv("finances.csv")
# df = df[["user_id", "spending", "date"]]
@app.route('/')
def index():
    return "hi"
    
@app.route('/predict', methods=['GET'])
def predict():
    # connect to the database
    try:
        db = pymysql.connect(host="localhost",user="root",password='iloveyoub@e96',database='FinanceDB')
        print("Database connection success")
    except pymysql.Error as e:
        print("Database connection failure"+str(e))
    # query the data
    try:
        userid = request.args.get('userid')
        query = "SELECT * FROM monthly_transaction where user_id = " + userid
        df = pd.read_sql(query, db)
    except Exception as e:
        print(e)
        print("Error: unable to fetch data")
        db.close()
        return
        
    # plot the time series
    # plt.plot(data["month"], data["amount"])
    # plt.xlabel("Month")
    # plt.ylabel("Spending")
    # plt.show()

    # close the connection
    db.close()
    # trans date type
    # extract info of month and year from the column
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    print(df)

    # grouped = df.groupby(["user_id", "month", "year"]).mean()
    # grouped = grouped.reset_index()
    # # sort the date by user_id, year, and month
    # grouped = grouped.sort_values(by=["user_id", "year", "month"])

    # model = pm.auto_arima(df["spending"], start_p=1, start_q=1,
    #                   test='adf',       # use adftest to find optimal 'd'
    #                   max_p=3, max_q=3, # maximum p and q
    #                   m=1,              # frequency of series
    #                   d=None,           # let model determine 'd'
    #                   seasonal=False,   # No Seasonality
    #                   start_P=0, 
    #                   D=0, 
    #                   trace=True,
    #                   error_action='ignore',  
    #                   suppress_warnings=True, 
    #                   stepwise=True)
    #print(model.summary())


    # create a list to store the prediction result
    predictions = {}
    # traverse
    for user_id in df['user_id'].unique():
        # time-series 
        user_data = df[df['user_id'] == user_id][['amount', 'date']]
        monthsPrediction = {}
        user_data = user_data.set_index('date')

        # use the grid search over the range of parameter values
        p_values = range(0, 3)
        d_values = range(0, 3)
        q_values = range(0, 3)
        parameters = {'p': p_values, 'd': d_values, 'q': q_values}
        param_combinations = list(itertools.product(*parameters.values()))

        results = []
        for params in param_combinations:
            try:
                # test and fit each param to find the best
                model = ARIMA(user_data['amount'], order=params)
                model = model.fit()
                rmse = np.sqrt(mean_squared_error(user_data['amount'], model.fittedvalues))
                results.append((params, rmse))
            except:
                continue

        if results:
            # find the combination of parameter values with the lowest RMSE
            best_params, best_rmse = min(results, key=lambda x: x[1])
            print("Best parameter values:", best_params)
            print("Best RMSE:", best_rmse)
        else:
            # default parameter values 
            print("No valid ARIMA models found")
            best_params = (1, 1, 1)

        # train the model with the best parameter values and make predictions
        model = ARIMA(user_data['amount'], order=best_params)
        model = model.fit()
        # predict
        # next_month_prediction = model_fit.forecast()
        yhat = model.forecast(12)
        monthsPrediction = {i+1: float(y) for i, y in enumerate(yhat)}
        # store the result
        predictions[str(user_id)] = monthsPrediction
    # result in json type
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
