import re

print("Welcome to the Precious Metal Market Predictor Application. Use Ctrl^C to exit at any time.")
while True:
    metal = input("Pick a metal market (gold or platnium): ")
    metal = metal.lower()
    assert metal in {"gold", "platnium"}

    startTrain = input("Enter a start date for the training data interval (YYYY-MM-DD): ")
    assert re.match('[0-9][0-9][0-9][0-9]-[01][0-9]-[0123][0-9]')
    assert int(startTrain[0:4]) in range(1991, 2004)
    
    endTrain = input("Enter an end date for the training data interval (YYYY-MM-DD): ")
    assert re.match('[0-9][0-9][0-9][0-9]-[01][0-9]-[0123][0-9]')
    assert int(startTrain[0:4]) in range(1991, 2004)

    startTrain = input("Enter a start date for the training data interval (YYYY-MM-DD): ")
    assert re.match('[0-9][0-9][0-9][0-9]-[01][0-9]-[0123][0-9]')
    assert int(startTrain[0:4]) in range(1991, 2004)

    startTrain = input("Enter a start date for the training data interval (YYYY-MM-DD): ")
    assert re.match('[0-9][0-9][0-9][0-9]-[01][0-9]-[0123][0-9]')
    assert int(startTrain[0:4]) in range(1991, 2004)


    print('Loading...')
    ### train models and display graphs here
    print('Done!')