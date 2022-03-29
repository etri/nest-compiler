import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt 
from sklearn.metrics import pairwise
import time
import argparse
import data

parser = argparse.ArgumentParser(description='XGBoost Rank')
parser.add_argument('--csv_path', type=str, required=True, help='Setting data path (default: data)')
parser.add_argument('--train_num', type=int, default='1000', metavar='N', help='Number of data to use for training (default: 1000)')
parser.add_argument('--update_num', type=int, default='5', metavar='N', help='How many data to update when learning online (default: 5)')
parser.add_argument('--random_num', type=int, default='100', metavar='N', help='Number of data to perform random updates (default: 100)')
parser.add_argument('--method', type=str, default='data', choices=['data', 'model', 'data+model'], help='Choose online learning method (default: data)')
args = parser.parse_args()

def update_pred_distance(pred, Y_test, list):
  pred_distances = pairwise.euclidean_distances(pred, Y_test)
  pred_result = pred_distances.mean()
  list.append(pred_result)
  return list

def update_min_value(pred, X_data, list):
  pred_min_index = np.argmin(pred.astype(int))
  pred_min = X_data[pred_min_index]
  list.append(pred_min)
  return list, pred_min

def update_pred_min_data(pred_min, list):
  if (len(list) == 0):
    list.append(pred_min)
  else :
    if  list[-1] > pred_min :
      list.append(pred_min)
    else:
      list.append(list[-1])
  return list

params = {'objective': 'rank:pairwise',
          'learning_rate':0.1,
          'max_depth':6,
          'gamma':0, 
          'reg_alpha':0.01,
          'subsample':0.79,
          'colsample_bytree':0.9}

list_distances = []
list_min = []
list_min_update = []
list_top1_num = []
data_num = args.update_num # update data num

file_path = args.csv_path
print(file_path[7:-4])

df =pd.read_csv(file_path)
X_train, Y_train, X_test, Y_test, X_data = data.data_setting('rank', df)

unexplode_X = X_train[data_num:]
unexplode_Y = Y_train[data_num:]
explode_X = X_train[:data_num]
explode_Y = Y_train[:data_num]

xg_train = xgb.DMatrix(explode_X, label=explode_Y)
xg_test = xgb.DMatrix(X_test, label=Y_test)
xg_untrain = xgb.DMatrix(unexplode_X, label=unexplode_Y)

watchlist  = [(xg_train,'train'),(xg_test,'eval')]

start = time.time()

model = xgb.train(params, xg_train, 200)

for i in range (data_num,args.train_num+data_num,data_num):
  print(i)
  
  if args.method == 'data':
    model = xgb.train(params, xg_train, 300, watchlist, early_stopping_rounds = 50)
  else:
    model = xgb.train(params, xg_train, 300, watchlist, early_stopping_rounds = 50, xgb_model=model)
  
  pred = model.predict(xg_test) 
  pred = np.reshape(pred,Y_test.shape)

  list_distances = update_pred_distance(pred, Y_test, list_distances)
  list_min, pred_min = update_min_value(pred, X_data, list_min)
  list_min_update = update_pred_min_data(pred_min, list_min_update)

  r_pred = model.predict(xg_untrain)
  r_pred = r_pred.astype(int)

  if (i < args.random_num): # random update 
    pred_index = np.where((r_pred == 1))[0]
    list_top1_num.append(len(pred_index))
    explode_X = np.append(explode_X, unexplode_X[:data_num], axis=0)
    explode_Y = np.append(explode_Y, unexplode_Y[:data_num], axis=0)
    unexplode_X = unexplode_X[data_num:]
    unexplode_Y = unexplode_Y[data_num:]
  else : # update the min predict
    pred_index = np.where((r_pred == 1))[0] 
    list_top1_num.append(len(pred_index))
    explode_X = np.append(explode_X, unexplode_X[pred_index[:5]], axis=0)
    explode_Y = np.append(explode_Y, unexplode_Y[pred_index[:5]], axis=0)
    np.delete(unexplode_X,(pred_index))
    np.delete(unexplode_Y,(pred_index))

  xg_untrain = xgb.DMatrix(unexplode_X, label=unexplode_Y)

  if args.method == 'model':
    xg_train = xgb.DMatrix(explode_X[-5:], label=explode_Y[-5:])
  else:
    xg_train = xgb.DMatrix(explode_X, label=explode_Y)

exe_t = time.time() - start
print("time :", exe_t)

print(np.array(list_distances).shape)
print(np.array(list_min).shape)
print(np.array(list_min_update).shape)
# save csv
result_csv = np.dstack([np.array(list_distances), np.array(list_min).reshape(-1)])
result_csv = np.dstack([result_csv, np.array(list_min_update).reshape(-1)])
result_csv = np.dstack([result_csv, np.array(list_top1_num)])

print(result_csv.reshape(-1,4))
pd.DataFrame(result_csv.reshape(-1,4)).to_csv(file_path[7:-4]+'_rank_' + args.method + '_update_'+str(exe_t)+".csv")

# save figure
vaild_num = len(X_data)-len(np.where(X_data == 20)[0])
cut_num = vaild_num // 100
cut_line_value = np.sort(X_data, axis=0)[cut_num]

num = ((np.where(np.array(list_min)<cut_line_value)[0])*data_num)
x = list(range(data_num,args.train_num+data_num,data_num))

fig = plt.figure(figsize=(15, 15))
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)

ax1.plot(x, list_distances, '-')
ax1.set_ylabel('pairwise')
ax1.set_title(str(cut_line_value) +' >= trial : ' + str(num) +", min value: " + str(list_min_update[-1]))

ax2.plot(x, list_min, '-')
ax2.set_ylabel('latency')
ax2.axhline(y=cut_line_value, color='gray', linestyle='--', linewidth=1)

ax3.plot(x, list_min_update, '-')
ax3.set_ylabel('latency')
ax3.axhline(y=cut_line_value, color='gray', linestyle='--', linewidth=1)

ax3.plot(x, list_top1_num, '-')
ax3.set_xlabel('data')
ax3.set_ylabel('top1 num')
ax3.axhline(y=5, color='gray', linestyle='--', linewidth=1)

plt.savefig(file_path[7:-4]+'_rank_' + args.method + '_update_'+str(exe_t)+'.png')
