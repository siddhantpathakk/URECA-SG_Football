from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import scripts.constants as constants


class DecTree_ML:
    """
    Class for Decision Tree modelling using ML (scikit-learn)
    """

    def __init__(self, _Xtrain, _ytrain, _Xtest, _ytest):
        """
        Initialisation function (by default)
        """
        self.X_train, self.y_train, self.X_test, self.y_test = (
            _Xtrain,
            _ytrain,
            _Xtest,
            _ytest,
        )
        self.y_pred, self.probs = None, []
        # print("The model has been initialised.")

    def fit(self):
        """
        Fit the Logistic Regression model (from sklearn) on X_train and y_train
        """
        model = DecisionTreeClassifier(max_depth=constants.MAX_DEPTH)
        model.fit(self.X_train, self.y_train.values.ravel())
        # print("The model has been fit.")
        return model

    def predict(self, PATH_TO_STORE):
        """
        Make predictions using the Logistic Regression model on X_test
        """
        model = self.fit()
        self.y_pred = model.predict(self.X_test)  # store binary output
        pred_prob = model.predict_proba(
            self.X_test
        )  # sklearn function to get probs instead of 0,1

        self.probs = []  # store probabilsitic output
        for i in pred_prob:
            c1, c2 = i[0], i[1]
            if c1 > c2:
                self.probs.append((c1, "Not a goal"))
            else:
                self.probs.append((c2, "Goal"))

        # print("The model has successfully predicted the values for test set.")

        # Train and test set accuracy
        print(
            "Accuracy of model on train set: "
            + str(model.score(self.X_train, self.y_train))
        )
        print(
            "Accuracy of model on test set: "
            + str(model.score(self.X_test, self.y_test))
        )

        """
        Calculates accuracy, confusion matrix, classification report and draws the ROC-AUC curve
        Input: a path to the directory where the graph should be stored
        """

        # Confusion Matrix
        from sklearn.metrics import confusion_matrix

        confusion_matrix = confusion_matrix(self.y_test, self.y_pred)
        print("The confusion matrix is as follows: ")
        print(confusion_matrix)

        # Classification Report
        from sklearn.metrics import classification_report

        print("\nThe classification report is as follows: ")
        print(classification_report(self.y_test, self.y_pred))

        # ROC-AUC
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import roc_curve

        logit_roc_auc = roc_auc_score(self.y_test, model.predict(self.X_test))
        fpr, tpr, thresholds = roc_curve(
            self.y_test, model.predict_proba(self.X_test)[:, 1]
        )
        plt.figure()
        plt.plot(fpr, tpr, label="Decision Tree (area = %0.2f)" % logit_roc_auc)
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic")
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(PATH_TO_STORE + "/ROC_dt.png")
        # plt.show()
        return tpr, fpr
