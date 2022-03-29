import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt 
from sklearn.metrics import mean_squared_error
import time
import argparse
import data

parser = argparse.ArgumentParser(description='XGBoost Regression')
parser.add_argument('--csv_path', type=str, required=True, help='Setting data path (default: data)')
parser.add_argument('--train_num', type=int, default='1000', metavar='N', help='Number of data to use for training (default: 1000)')
parser.add_argument('--update_num', type=int, default='5', metavar='N', help='How many data to update when learning online (default: 5)')
parser.add_argument('--random_num', type=int, default='100', metavar='N', help='Number of data to perform random updates (default: 100)')
parser.add_argument('--method', type=str, default='data', choices=['data', 'model', 'data+model'], help='Choose online learning method (default: data)')
args = parser.parse_args()

def update_rmse(pred, Y_test, list):
  pred_rmse = mean_squared_error(pred, Y_test)
  list.append(pred_rmse)
  return list

def update_min_value(pred, Y_test, list):
  pred_min_index = np.argmin(pred)
  pred_min = Y_test[pred_min_index]
  list.append(pred_min)
  return list, pred_min

def update_min_data_num(pred, list):
  min_num = np.where(pred == np.min(pred))[0].tolist()
  list.append(len(min_num))
  return list

def update_pred_min(pred_min, list):
  if (len(list) == 0):
    list.append(pred_min)
  else :
    if list[-1] > pred_min :
      list.append(pred_min)
    else:
      list.append(list[-1])
  return list

params = {'objective': 'reg:squarederror',
          'learning_rate':0.1,
          'max_depth':6,
          'gamma':0, 
          'reg_alpha':0.01,
          'subsample':0.79,
          'colsample_bytree':0.9}

data_num = args.update_num # update data num

list_rmse = []
list_min = []
list_min_update = []
list_min_num = []

file_path = args.csv_path
print(file_path[7:-4])

df =pd.read_csv(file_path)
X_train, Y_train, X_test, Y_test = data.data_setting('err_nspilt', df)

explode_X = X_train[:data_num] # already trained data
explode_Y = Y_train[:data_num]
unexplode_X = X_train[data_num:] # data to learn
unexplode_Y = Y_train[data_num:]

xg_train = xgb.DMatrix(explode_X, label=explode_Y)
xg_test = xgb.DMatrix(X_test, label=Y_test)

watchlist  = [(xg_train,'train'),(xg_test,'eval')]

start = time.time() # check train time

model = xgb.train(params, xg_train, 200, watchlist, early_stopping_rounds = 30)

for i in range (data_num,args.train_num+data_num,data_num):
  print(i)
  
  if args.method == 'data':
    model = xgb.train(params, xg_train, 300, watchlist, early_stopping_rounds = 50)
  else:
    model = xgb.train(params, xg_train, 300, watchlist, early_stopping_rounds = 50, xgb_model=model)

  pred = model.predict(xg_test)

  list_rmse = update_rmse(pred, Y_test, list_rmse)
  list_min, pred_min = update_min_value(pred, Y_test, list_min)
  list_min_num = update_min_data_num(pred, list_min_num)
  list_min_update = update_pred_min(pred_min, list_min_update)

  if (i < args.random_num): # random update 
    explode_X = np.append(explode_X, unexplode_X[:data_num], axis=0)
    explode_Y = np.append(explode_Y, unexplode_Y[:data_num], axis=0)
    unexplode_X = unexplode_X[data_num:]
    unexplode_Y = unexplode_Y[data_num:]
  else : # update the min predict
    pred_index = np.argpartition(pred, data_num)[:data_num]
    explode_X = np.append(explode_X, unexplode_X[pred_index], axis=0)
    explode_Y = np.append(explode_Y, unexplode_Y[pred_index], axis=0)
    np.delete(unexplode_X,(pred_index))
    np.delete(unexplode_Y,(pred_index))

  if args.method == 'model':
    xg_train = xgb.DMatrix(explode_X[-5:], label=explode_Y[-5:])
  else:
    xg_train = xgb.DMatrix(explode_X, label=explode_Y)

exe_t = time.time() - start
print("time :", exe_t)

# save csv
result_csv = np.dstack([np.array(list_rmse), np.array(list_min).reshape(-1)])
result_csv = np.dstack([result_csv, np.array(list_min_update).reshape(-1)])
result_csv = np.dstack([result_csv, np.array(list_min_num)])

pd.DataFrame(result_csv.reshape(-1,4)).to_csv(file_path[7:-4]+'_reg_' + args.method + '_update_'+str(exe_t)+".csv")

# save figure
vaild_num = len(Y_test)-len(np.where(Y_test == 20)[0])
cut_num = vaild_num // 100
cut_line_value = np.sort(Y_test, axis=0)[cut_num]

num = ((np.where(np.array(list_min)<=cut_line_value)[0])*data_num)

x = list(range(data_num,args.train_num+data_num,data_num))

print(len(list_rmse))

fig = plt.figure(figsize=(15, 15))
ax1 = fig.add_subplot(4, 1, 1)
ax2 = fig.add_subplot(4, 1, 2)
ax3 = fig.add_subplot(4, 1, 3)
ax4 = fig.add_subplot(4, 1, 4)

ax1.plot(x, list_rmse, '-')
ax1.set_ylabel('rmse')
ax1.set_title(str(cut_line_value) +' >= trial : ' + str(num) +", min value: " + str(list_min_update[-1]))

ax2.plot(x, list_min, '-')
ax2.set_ylabel('latency')
ax2.axhline(y=cut_line_value, color='gray', linestyle='--', linewidth=1)

ax3.plot(x, list_min_update, '-')
ax3.set_ylabel('latency')
ax3.axhline(y=cut_line_value, color='gray', linestyle='--', linewidth=1)

ax4.plot(x, list_min_num, '-')
ax4.axis(ymin=0,ymax=100)
ax4.set_xlabel('data')
ax4.set_ylabel('min num')
ax4.axhline(y=2, color='gray', linestyle='--', linewidth=1)

plt.savefig(file_path[7:-4]+'_reg_' + args.method + '_update_'+str(exe_t)+'.png')
