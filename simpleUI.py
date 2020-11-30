import re

print("Welcome to the Precious Metal Market Predictor Application. Use Ctrl^C to exit at any time.")
while True:
    metal = input("Pick a metal market (gold or platnium): ")
    metal = metal.lower()
    assert metal in {"gold", "platnium"}

    startTrain = input("Enter a start date for the training data interval (YYYY-MM-DD): ")
    assert re.match('[0-9][0-9][0-9][0-9]-[01][0-9]-[0123][0-9]', startTrain) is not None
    assert int(startTrain[0:4]) in range(1991, 2004)
    
    endTrain = input("Enter an end date for the training data interval (YYYY-MM-DD): ")
    assert re.match('[0-9][0-9][0-9][0-9]-[01][0-9]-[0123][0-9]', endTrain) is not None
    assert int(startTrain[0:4]) in range(1991, 2004)

    startPred = input("Enter a start date for the prediction data interval (YYYY-MM-DD): ")
    assert re.match('[0-9][0-9][0-9][0-9]-[01][0-9]-[0123][0-9]', startPred) is not None
    assert int(startTrain[0:4]) in range(1991, 2008)

    endPred = input("Enter an end date for the prediction data interval (YYYY-MM-DD): ")
    assert re.match('[0-9][0-9][0-9][0-9]-[01][0-9]-[0123][0-9]', endPred) is not None
    assert int(startTrain[0:4]) in range(1991, 2008)


    print('Loading...')
    ### train models and display graphs here
    ## Import all the needed machine learning libraries
    

    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    import sklearn as sk
    import quandl

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.datasets import make_regression
    from keras.models import Sequential
    from keras.layers import Dense


      
    plt.style.use("ggplot")
    ## Import all of the metal data, and format the datafram for training 

    quandl.ApiConfig.api_key = '-sB9AdohxNUey-w8-99z'
    sp500 = quandl.get("MULTPL/SP500_PE_RATIO_MONTH",start_date='1991-01-01', end_date='2008-7-31' )
    metals = quandl.get('LBMA/GOLD',start_date='1991-01-01', end_date='2008-7-31')
    metals = metals.drop(columns = ['EURO (AM)','EURO (PM)','GBP (AM)', 'USD (PM)','GBP (PM)'])
    plat=quandl.get("LPPM/PLAT",start_date='1991-01-01', end_date='2008-7-31')
    plat =plat.drop(columns = ['EUR AM','EUR PM','GBP AM', 'USD PM','GBP PM'])
    sp500['SMA'] = sp500.iloc[:,0].rolling(window = 3).mean()-sp500.iloc[:,0]
    metals['PLAT'] = plat['USD AM']
    metals['Index'] = range(1,4446)
    metals['dates'] = pd.date_range(start='1991-01-01', periods= 4445)
    metals= metals.set_index('Index')

    startTrain2 = metals[metals['dates']==startTrain].index.values
    startTrain2 = startTrain2[0]
    endTrain2 = metals[metals['dates']==endTrain].index.values
    endTrain2 = endTrain2[0]
    startPred2 = metals[metals['dates']==startPred].index.values
    startPred2 = startPred2[0]
    endPred2 = metals[metals['dates']==endPred].index.values
    endPred2 = endPred2[0]


   

    ## set the X and Y values off of the selcted metal 
    if metal == 'gold':
        X_train=metals.iloc[startTrain2:endTrain2,1].values[:,np.newaxis]
        y_train=metals.iloc[startTrain2:endTrain2,0].values[:,np.newaxis]

        X_test = metals.iloc[startPred2:endPred2,1].values[:,np.newaxis]
        y_test = metals.iloc[startPred2:endPred2,0].values[:,np.newaxis]

        Xten=metals.iloc[startTrain2:endTrain2,[0,1]].values
        yten=metals.iloc[startTrain2:endTrain2,0].values
        x_tentest= metals.iloc[startPred2:endPred2,[0,1]].values

    else:
        y_train=metals.iloc[startTrain2:endTrain2,1].values[:,np.newaxis]
        X_train=metals.iloc[startTrain2:endTrain2,0].values[:,np.newaxis]  

        y_test = metals.iloc[startPred2:endPred2,1].values[:,np.newaxis]
        X_test = metals.iloc[startPred2:endPred2,0].values[:,np.newaxis]

        yten=metals.iloc[startTrain2:endTrain2,[0,1]].values
        Xten=metals.iloc[startTrain2:endTrain2,0].values
        x_tentest= metals.iloc[startPred2:endPred2,[0,1]].values
    
    # plot the training and test data 

    plt.scatter(X_train, y_train, s = 5)
    plt.scatter(X_test,y_test, s= 5 )
    plt.show()

    #Traing the linear model and the deep learning model

    regr = MLPRegressor(random_state=2, max_iter=1000,solver = 'lbfgs',alpha= .01).fit(X_train, y_train)
    y_linear = regr.predict(X_test)

    model = Sequential()
    model.add(Dense(20, activation="relu", input_dim=2, kernel_initializer="uniform"))
    model.add(Dense(1, activation="linear", kernel_initializer="uniform"))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    model.fit(Xten,yten, epochs=50, batch_size=10,  verbose=2)

    y_tensor = model.predict(x_tentest)
    #Plot the results of each model
    #dates = metals.iloc[startPred2:endPred2,2].values[:,np.newaxis]
    dates = np.arange(startPred2,endPred2,1)


    plt.plot(dates,y_test)
    plt.plot(dates,y_linear)
    plt.plot(dates,y_tensor)
    plt.show()


    plt.scatter(X_test,y_test, s = 5)
    plt.plot(X_test,y_linear)
    plt.show()

    plt.scatter(X_test,y_test, s = 5)
    plt.plot(X_test,y_tensor)
    plt.show()



    print('Done!')