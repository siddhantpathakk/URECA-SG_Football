import argparse, sys, os, os.path, tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from datetime import date
import logreg_dl, scripts.ml.logreg_ml as logreg_ml
import scripts.ml.dectree_ml as dectree_ml, scripts.ml.linearsvc_ml as linearsvc_ml
import scripts.ml.nusvc_ml as nusvc_ml, scripts.ml.knn_ml as knn_ml  # All the local python files
import scripts.dataLoader as dataLoader
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an ML/DL model"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="all",
        help="choice of model",
    )
    parser.add_argument(
        "--level",
        type=int,
        required=False,
        default=1,
        help="level of information (1-5)",
    )
    args = parser.parse_args()
    lvl = args.level
    # Logic to get the date and store console logs as .txt
    today = date.today()
    d4 = today.strftime("%b-%d-%Y")
    # _p = "Inferences/Result Logs/" + d4
    _p = "Inferences/Result Logs/" + d4+"/Level "+str(lvl)

    
    if os.path.exists(_p) == False:
        os.mkdir(_p)
        print(
            "The result directory for "
            + d4
            + " has been created, so console outputs will be saved at "
            + _p
            + "/results_"+ args.model+ ".txt"
        )
    else:
        print("The result directory for "
            + d4
            + " already exists, so console outputs will be saved there")

    # Start storing the console logs onto the .txt file
    if args.model !="all":
        result_path = _p + "/results_" + args.model + ".txt"
        sys.stdout = open(result_path, "w")


    if lvl==1:
        col_list = ['ScoreHomeTeam', 'ScoreAwayTeam', 'timeLeft', 'LocationX', 'LocationY','HomeTeam_no','HomeTeam_yes','PhaseType_OpenPlay', 'PhaseType_SetPlay' ]
    
    if lvl==2:
        col_list = ['ScoreHomeTeam', 'ScoreAwayTeam', 'timeLeft','LocationX', 'LocationY','HomeTeam_no','HomeTeam_yes']

    if lvl==3:
        col_list = ['ScoreHomeTeam', 'ScoreAwayTeam', 'timeLeft', 'LocationX', 'LocationY']
    
    if lvl==4:
        col_list = ['ScoreHomeTeam', 'ScoreAwayTeam', 'LocationX', 'LocationY']

    if lvl==5:
        col_list = ['LocationX', 'LocationY']
    
    np.random.seed(20230322)
    tf.random.set_seed(20230322)

    paths = dataLoader.combine_all_paths()
    dataframe = dataLoader.create_dataframe(paths)
    # print(dataframe)
    # print(len(paths))
    # print(paths[0])
    # print(dataframe.columns)
    # dataframe = dataframe.drop(columns=["Unnamed: 0"],axis=1)
    Data_ohe = dataLoader.encode_dataframe(dataframe)
    X,y = dataLoader.split_into_Xy(Data_ohe)
    # print(X.columns)
    X, y = dataLoader.resample_data(X, y)
    # X = dataLoader.normalize_X(X)
    X = X[col_list]
    # print(X.columns)
    # # perform the EDA => TO BE COMPLETED
    # process.analyze_data(X, y)

    # convert the data into X_train, y_train, X_test, y_test
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=12345
    )
    
    print(
        "\nFOR TRAINING:\n\tThere are "
        + str(len(X_train))+ " many data points and " + str(len(y_train)) + " many desired outputs.")
    
    print(
        "\t"
        + str(y_train.value_counts()[0]) + " for no goal(0) v/s "+ str(y_train.value_counts()[1]) + " for goal(1)")

    print(
        "\nFOR VALIDATION:\n\tThere are "
        + str(len(X_test))+ " many data points and "
        + str(len(y_test)) + " many desired outputs.")

    print(
        "\t"
        + str(y_test.value_counts()[0])
        + " for no goal(0) v/s "
        + str(y_test.value_counts()[1])
        + " for goal(1)"
    )

    if args.model=='all':
        result_path = _p + "/results_lr.txt"
        sys.stdout = open(result_path, "w")
        logreg = logreg_ml.LogReg_ML(X_train, y_train, X_test, y_test)  # load
        tlr, flr = logreg.predict(PATH_TO_STORE = _p)  # fit, predict, accuracy
        sys.stdout.close()

        result_path = _p + "/results_dt.txt"
        sys.stdout = open(result_path, "w")
        dectree = dectree_ml.DecTree_ML(X_train, y_train, X_test, y_test)  # load
        tdt, fdt = dectree.predict(PATH_TO_STORE = _p)  # fit, predict, accuracy
        sys.stdout.close()

        result_path = _p + "/results_linearsvc.txt"
        sys.stdout = open(result_path, "w")       
        dectree = linearsvc_ml.LinearSVC_ML(X_train, y_train, X_test, y_test)  # load
        tlsvc, flsvc = dectree.predict(PATH_TO_STORE = _p)  # fit, predict, accuracy
        sys.stdout.close()
       
        result_path = _p + "/results_nusvc.txt"
        sys.stdout = open(result_path, "w")
        dectree = nusvc_ml.NuSVC_ML(X_train, y_train, X_test, y_test)  # load
        tnu, fnu = dectree.predict(PATH_TO_STORE = _p)  # fit, predict, accuracy
        sys.stdout.close()

        result_path = _p + "/results_knn.txt"
        sys.stdout = open(result_path, "w")
        dectree = knn_ml.KNN_ML(X_train, y_train, X_test, y_test)  # load
        tknn, fknn = dectree.predict(PATH_TO_STORE = _p)  # fit, predict, accuracy
        sys.stdout.close()
        
        # FPR and TPR values for each model
        model_names = ["Logistic Regression",
               "K-Nearest Neighbors", 
               "Nu SVC", 
               "Decision Tree",
                "Linear SVC", 
               ]
        
        fpr_values = [
                flr,
                fknn,
                fnu,
                fdt,
                flsvc,
                ]
        tpr_values = [
            tlr,
            tknn,
            tnu,
            tdt,
            tlsvc,
        ]

        # Plotting the curves
        plt.figure(figsize=(10, 10))
        
        # new figure object
        
        # plt.figure(figsize=(10, 10))

        for fpr, tpr, model_name in zip(fpr_values, tpr_values, model_names):
            plt.plot(fpr, tpr, label=model_name)
            # print(model_name, "done...")

        plt.plot([0, 1], [0, 1], "r--")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        # plt.show()
        plt.savefig("ROC_lvl"+str(lvl)+".png")


    # if arg = ML => load model, fit, predict, accuracy => store logs
    if args.model == "lr":
        logreg = logreg_ml.LogReg_ML(X_train, y_train, X_test, y_test)  # load
        logreg.predict(PATH_TO_STORE = _p)  # fit, predict, accuracy
        # Stop storing the console logs onto the .txt file
        sys.stdout.close()
        
    # if arg = DL => create model, compile, fit, accuracy => store logs
    elif args.model == "nn":
        os.system('python nn_finetuned.py')
        # Stop storing the console logs onto the .txt file
        sys.stdout.close()
        
    # if arg = ML => load model, fit, predict, accuracy => store logs
    elif args.model == "dt":
        dectree = dectree_ml.DecTree_ML(X_train, y_train, X_test, y_test)  # load
        dectree.predict(PATH_TO_STORE = _p)  # fit, predict, accuracy
        # Stop storing the console logs onto the .txt file
        sys.stdout.close()
        
    # if arg = ML => load model, fit, predict, accuracy => store logs
    elif args.model == "linearsvc":
        dectree = linearsvc_ml.LinearSVC_ML(X_train, y_train, X_test, y_test)  # load
        dectree.predict(PATH_TO_STORE = _p)  # fit, predict, accuracy
        # Stop storing the console logs onto the .txt file
        sys.stdout.close()
        
    # if arg = ML => load model, fit, predict, accuracy => store logs
    elif args.model == "nusvc":
        dectree = nusvc_ml.NuSVC_ML(X_train, y_train, X_test, y_test)  # load
        dectree.predict(PATH_TO_STORE = _p)  # fit, predict, accuracy
        # Stop storing the console logs onto the .txt file
        sys.stdout.close()
        
    # if arg = ML => load model, fit, predict, accuracy => store logs
    elif args.model == "knn":
        dectree = knn_ml.KNN_ML(X_train, y_train, X_test, y_test)  # load
        dectree.predict(PATH_TO_STORE = _p)  # fit, predict, accuracy

        # Stop storing the console logs onto the .txt file
        sys.stdout.close()
