import sys, os
sys.path.append('c:/Users/Administrator/dr박연수/')
from config import *
import utils.visualization as vis
import pandas as pd
import glob
import re

def make_df(csv, val, train):
    df = pd.read_csv(version_dir+'/'+csv)
    df = df.drop('Fold', axis=1)
    df_loss = df[['train_loss', 'val_loss']]
    df_others = df.drop(['train_loss', 'val_loss'], axis=1)

    if val:
        df_loss = df_loss.drop([d for d in df_loss.columns if d.startswith('train')], axis=1)
        df_others = df_others.drop([d for d in df_others.columns if d.startswith('train')], axis=1)
    
    elif train:
        df_loss = df_loss.drop([d for d in df_loss.columns if d.startswith('val')], axis=1)
        df_others = df_others.drop([d for d in df_others.columns if d.startswith('val')], axis=1)

    return df, df_loss, df_others
    

def print_df(df):
    for col in df.columns:
        print(col)
        df.iloc[len(df)-2]
        df.iloc[len(df)-1]


def print_info(title_list, test_columns):
    print("================ the best result ================")
    print('minimum loss mean title  : ', title_list[0])
    print('maximum others mean title: ', title_list[1])
    print('minimum loss std title   : ', title_list[2])
    print('minimum others std title : ', title_list[3])
    print("=================================================")
    result = []

    for value in title_list:
        if value not in result:
            result.append(value)

    for li in result:
        df, _, _ = make_df(li, val=0, train=0)
        temp = df[test_columns].iloc[len(df)-2:len(df)]
        temp.index = ['mean','std']
        print(li)
        print(temp)
        print()
        
def print_performance_each_task(csv_list):
    min_loss_mean   = float('inf')
    max_others_mean = float('-inf')
    min_loss_std    = float('inf')
    min_others_std  = float('inf')

    min_loss_mean_title   = str()
    max_others_mean_title = str()
    min_loss_std_title    = str()
    min_others_std_title  = str()

    val   = 0
    train = 0

    # print(f"val : {val} train : {train}")
    for csv in csv_list:
        df, df_loss, df_others = make_df(csv, val, train)
        
        # 'test'가 포함된 열 이름 필터링
        test_columns = [col for col in df.columns if 'test' in col]
        
        loss_mean   = sum(df_loss.iloc[len(df)-2])
        others_mean = sum(df_others.iloc[len(df)-2]) 
        loss_std    = sum(df_loss.iloc[len(df)-1])
        others_std  = sum(df_others.iloc[len(df)-1])
        
        if loss_mean < min_loss_mean:
            min_loss_mean_title = csv
            min_loss_mean = loss_mean

        if others_mean > max_others_mean:    
            max_others_mean_title = csv
            max_others_mean = others_mean

        if loss_std < min_loss_std:
            min_loss_std_title = csv
            min_loss_std = loss_std

        if others_std < min_others_std:
            min_others_std_title = csv
            min_others_std = others_std

    title_list = [min_loss_mean_title, 
                max_others_mean_title, 
                min_loss_std_title, 
                min_others_std_title]

    print_info(title_list, test_columns)

def sort_orderby_columns(column_name, csv_files):
    # 모든 CSV 파일 경로 가져오기
    print(f'========{column_name}========')

    # CSV 파일과 그 안의 test_column 값 저장
    file_column = []

    for file in csv_files:
        try:
            # CSV 파일 읽기
            df, df_loss, df_others = make_df(file, 0, 0)
            
            # column_name 열이 있는지 확인
            if column_name in df.columns:
                # test_column 값을 가져오고 평균으로 정렬
                mean_value = df[column_name].iloc[len(df)-2]  # 또는 적절한 값을 선택
                # file_column.append((file, column_value))
                max_value = df[column_name].max()
                file_column.append((file, max_value, mean_value))  # 파일 이름과 가장 큰 값 저장

        except Exception as e:
            print(f"Error reading {file}: {e}")
    # test_column 값에 따라 오름차순으로 정렬
    file_column.sort(key=lambda x: x[1])

    # 결과 출력
    for file, max_value, mean_value in file_column:
        print(f"File: {file}    ,   max_{column_name}: {max_value},   mean_{column_name}: {mean_value}")

def make_result(file_name, df):
    for file_name in task_2_list:
        result_df, _, _ = make_df(file_name, 0, 0)

        tile_match = re.search(r'Tile(\d+)', file_name)
        crop_task_match = re.search(r'Crop(\w+)', file_name)
        aug_match = re.search(r'aug(True|False)', file_name)
        task_match = re.search(r'task(\d+)', file_name)

        tile_value = tile_match.group(1) if tile_match else "Unknown"
        crop_task_value = crop_task_match.group(1) if crop_task_match else "Unknown"
        aug_value = aug_match.group(1) if aug_match else "Unknown"
        task_value = task_match.group(1) if task_match else "Unknown"

        if task_value != "Unknown":
            crop_value = crop_task_value.split('task')[0]
        else:
            crop_value = crop_task_value

        mean_aucroc = result_df['test_aucroc'].iloc[len(result_df)-2]
        mean_aucroc = f"{mean_aucroc:.4f}"

        mean_recall = result_df['test_recall'].iloc[len(result_df)-2]
        mean_recall = f"{mean_recall:.4f}"

        mean_precision = result_df['test_precision'].iloc[len(result_df)-2]
        mean_precision = f"{mean_precision:.4f}"

        mean_f1_score = result_df['test_f1_score'].iloc[len(result_df)-2]
        mean_f1_score = f"{mean_f1_score:.4f}"        
        data = {
                "tile size": [int(tile_value)],
                "crop": [crop_value],
                "augmentation": [aug_value],
                "avg. aucroc":[mean_aucroc],
                "avg. recall":[mean_recall],
                "avg. precision":[mean_precision],
                "avg. f1 score":[mean_f1_score],
            
            }

        df = pd.concat([pd.DataFrame(data), df])
    df = df.sort_values(by=['tile size', 'crop', 'augmentation', 'avg. aucroc'], ascending=True).reset_index(drop=True)

    return df

if __name__ == '__main__':
    print('now compare phase..')
    save_version = vis.get_save_version(save_dir)
    version_dir = os.path.join(save_dir, f'version_{save_version}/fold_result')
    csv_list = [df for df in os.listdir(version_dir) if df.endswith('.csv')]
    task_2_list = [] 
    task_3_list = []
    data = {
        "tile size": [],
        "crop": [],
        "augmentation": [],
        "avg. aucroc":[],    
        "avg. recall":[],
        "avg. precision":[],
        "avg. f1 score":[],
        
    }

    df = pd.DataFrame(data)

    for csv in csv_list:
        # 'task' 다음에 숫자가 바로 붙어 있는지 찾기
        tasks = re.findall(r'task(\d+)', csv)
        if tasks == ['2']:
            task_2_list.append(csv)
        else:
            task_3_list.append(csv)

    df = make_result(task_2_list, df)
    print(df)
    df.to_csv('result.csv')

    # print_performance_each_task(task_2_list)
    # print_performance_each_task(task_3_list)
    # sort_orderby_columns('test_aucroc',task_2_list)
    # sort_orderby_columns('test_accuracy',task_2_list)
