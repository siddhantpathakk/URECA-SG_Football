import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, warnings, sys, os, os.path
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, AlphaDropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from datetime import date
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)
from keras.layers import LeakyReLU
LeakyReLU = LeakyReLU(alpha=0.1)

#Load Model configuration
import json
with open('config.json') as json_file:
    params_nn_ = json.load(json_file)


today = date.today()
d4 = today.strftime("%b-%d-%Y")
lvl = 1
_p = "Inferences/Result Logs/" + d4+"/Level "+str(lvl)

if os.path.exists(_p) == False:
    os.mkdir(_p)

print("Working Directory is:\t",_p,'\n')



result_path_for_config = _p + "/results_dnn.txt"
sys.stdout = open(result_path_for_config, "w")


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
    
# Load dataset
trainSet = pd.read_csv('Dataset csv files/encoded-dataframe.csv').drop(columns=['Unnamed: 0'],axis=1)
# Feature generation: training data
train = trainSet.dropna(axis=0)
# train = pd.get_dummies(train)
# train validation split
X_train, X_val, y_train, y_val = train_test_split(train.drop(columns=['isGoal'], axis=0), train['isGoal'],test_size=0.2, random_state=12112022,stratify=train['isGoal'])
X_train  = X_train[col_list]
X_val = X_val[col_list]
X_val.to_csv("Dataset csv files/Validation-X.csv")


if params_nn_['activation']=='Leaky ReLU':
    act = LeakyReLU
else:
    act = params_nn_['activation']

# Model creation
nn = Sequential()
nn.add(Dense(params_nn_['neurons'], input_dim = len(X_train.columns), activation=act, kernel_initializer='he_normal'))
if params_nn_['normalization'] > 0.5:
    nn.add(BatchNormalization())
for i in range(params_nn_['layers1']):
    nn.add(Dense(params_nn_['neurons'], activation=act, kernel_initializer='he_normal'))
if params_nn_['dropout'] > 0.5:
    nn.add(Dropout(params_nn_['dropout_rate'], seed=123))
for i in range(params_nn_['layers2']):
    nn.add(Dense(params_nn_['neurons'], activation=act,kernel_initializer='he_normal'))
nn.add(Dense(1, activation='sigmoid'))

# Callback Setup
es = EarlyStopping(monitor='accuracy', mode='max', verbose=0, patience=20, restore_best_weights=True)
# es2 = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20, restore_best_weights=True)
mc = ModelCheckpoint(_p+'/Model Checkpoints/model-{epoch:03d}-{val_accuracy:02f}-{val_loss:02f}.h5', verbose=1,monitor='accuracy', save_best_only=True, mode='max')


nn.summary()
nn.compile(optimizer = params_nn_['optimizer'], loss = 'binary_crossentropy', metrics=["accuracy"])
hist = nn.fit(X_train, y_train, epochs=params_nn_['epochs'], batch_size=params_nn_['batch_size'], validation_split=0.2, shuffle=True, callbacks = [mc])

# Training Graphs (can also use tensorboard)
plt.plot(hist.history["accuracy"], label="Train")
plt.plot(hist.history["val_accuracy"], label="Validation")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.title("Model Accuracy")
plt.legend(loc="upper left")
plt.savefig(_p + "/model_accuracy_dnn.png")
plt.close()

plt.plot(hist.history["loss"], label="Train")
plt.plot(hist.history["val_loss"], label="Validation")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Model Loss")
plt.legend(loc="upper right")
plt.savefig(_p + "/model_loss_dnn.png")
sys.stdout.close()


result_path_for_config = _p +"/results_eval_dnn.txt"
sys.stdout = open(result_path_for_config, "w")

test_loss, test_acc = nn.evaluate(X_val, y_val)
print("\nTest loss:\t", test_loss)
print("Test accuracy:\t", test_acc)

path_for_model = _p +"/model_"+"{:.1f}".format(test_acc*100)+".h5"
nn.save(path_for_model)

sys.stdout.close()


