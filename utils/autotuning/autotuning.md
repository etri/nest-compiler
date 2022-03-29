## Auto-Tuning 실행 방법

### XGBoost를 통한 Auto-Tuning 과정
* XGBoost를 사용하여 profiling 결과로 얻은 .csv 파일을 가지고 Auto-Tuning을 진행한다.
* profiling 결과로 얻은 데이터로는 regression과 rank가 가능해 두 가지 경우 모두 구현하였다.
* 각 학습은 online learning으로 진행되며 아래와 같이 3가지 방법을 가지고 있고 모두 구현하였다.
    - 데이터 업데이트
    - 모델 업데이트
    - 데이터와 모델 모두 업데이트
* 학습의 결과로 학습 과정을 보여주는 그래프와 학습 과정 중 얻은 데이터가 저장된 .csv 파일을 얻을 수 있다.

1. XGBoost Regression 실행

~~~
python xgboost_reg.py --csv_path ./data/ResConv1Test_data.csv --train_num 1000 --update_num 5 --random_num 100 --method 'data+model'
~~~

2. XGBoost Rank 실행

~~~
python xgboost_rank.py --csv_path ./data/ResConv1Test_data.csv --train_num 1000 --update_num 5 --random_num 100 --method 'data+model'
~~~
