# cs5242-project, Qiu Hai, Zuo Shuman

0. Add training data and testing data and modify __CONST.py__ to specify 'training_data' and 'testing_data' directories
1. Preprocess data for training
    > run __data_preprocess.py__
2. Train models
    >run __main.py__ <br>
    All models at each epoch checkpoint will be saved to 'models' folder<br>
    Select a model and move both .h5 and .json file to 'selected_models' folder
    Then change MODEL.name in __CONST.py__ as selected(without .)
    this step will take long time 
3. Preprocess data for testing
    >run __data_preprocess_test.py__ <br>
    this step will produce large data files and take long time
4. Get 10 predictions for each protein
    > run test_prediction.py <br>
    this step will run about 

