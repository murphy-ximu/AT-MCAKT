import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from config import Config
from sklearn.model_selection import train_test_split
import gc
from time import *

# test figuration

"""
RAIEd2020
"""
def get_RAIEd_data():
    dtypes = {'timestamp': 'int64', 'user_id': 'int32', 'content_id': 'int16',
              'answered_correctly': 'int8', "content_type_id": "int8",
              "prior_question_elapsed_time": "float32", "task_container_id": "int16"}
    question_dtypes = {"question_id": "int16", "tags": "str", "tags_num": "int16"}
    print("loading csv.....")
    train_df = pd.read_csv(Config.TRAIN_FILE, usecols=[
        1, 2, 3, 4, 5, 7, 8], dtype=dtypes)
    # 题目
    question_df = pd.read_csv(Config.QUESTION_FILE, usecols=[
        0, 4, 5], dtype=question_dtypes)
    # Align the name of key column for latter merging
    question_df = question_df.rename(columns={"question_id": "content_id"})
    # 处理tags格式: str -> list[int]
    question_df.tags = question_df.tags.str.split(' ')
    for i in range(len(question_df.tags)):
        question_df.tags.at[i] = np.array(list(map(int, question_df.tags[i])), 'int64')
        question_df.tags.at[i] = np.pad(question_df.tags[i], (0, Config.MAX_CATS_PER - len(question_df.tags[i])),
                                        'constant', constant_values=(0))

    train_df = train_df[train_df.content_type_id == 0]
    # content_type_id:
    # 0 if the event was a question being posed to the user, 1 if the event was the user watching a lecture.
    train_df.prior_question_elapsed_time.fillna(0, inplace=True)
    train_df.prior_question_elapsed_time /= 3600
    # train_df.prior_question_elapsed_time.clip(lower=0,upper=300,inplace=True)
    train_df.prior_question_elapsed_time = train_df.prior_question_elapsed_time.astype(
        np.int32)
    # Merging train_df and question_df on
    train_df = train_df.merge(question_df, on='content_id', how="left")  # left outer join to consider part
    del question_df
    gc.collect()

    print("Computing question difficulty")
    df_difficulty = train_df["answered_correctly"].groupby(train_df["content_id"])
    train_df["popularity"] = df_difficulty.transform('size')
    train_df["difficulty"] = df_difficulty.transform('sum') / train_df["popularity"]
    print("Popularity max", train_df["popularity"].max(), ",Difficulty max", train_df["difficulty"].max())
    del df_difficulty
    gc.collect()

    print("shape of dataframe :", train_df.shape)

    # 时间排序 important!
    train_df = train_df.sort_values(
        ["timestamp"], ascending=True).reset_index(drop=True)
    n_questions = train_df.content_id.nunique()
    print("no. of questions :", n_questions)
    print("shape after exlusion:", train_df.shape)

    print("Calculating lag time")
    time_dict = {}
    lag_time_col = np.zeros(len(train_df), dtype=np.int64)
    for ind, row in enumerate(tqdm(train_df[["user_id", "timestamp", "content_id"]].values)):
        if row[0] in time_dict.keys():
            # if the task_container_id is the same, the lag time is not allowed
            if row[2] == time_dict[row[0]][1]:
                lag_time_col[ind] = time_dict[row[0]][2]
            else:
                timestamp_last = time_dict[row[0]][0]
                lag_time_col[ind] = row[1] - timestamp_last
                time_dict[row[0]] = (row[1], row[2], lag_time_col[ind])
        else:
            time_dict[row[0]] = (row[1], row[2], 0)
            lag_time_col[ind] = 0
        if lag_time_col[ind] < 0:
            raise RuntimeError("Has lag_time smaller than 0.")

    train_df["lag_time_s"] = lag_time_col // 1000
    train_df["lag_time_m"] = lag_time_col // (60 * 1000)
    train_df["lag_time_d"] = lag_time_col // (60 * 1000 * 1440)
    train_df.lag_time_s.clip(lower=0, upper=300, inplace=True)
    train_df.lag_time_m.clip(lower=0, upper=1440, inplace=True)
    train_df.lag_time_d.clip(lower=0, upper=365, inplace=True)
    train_df.lag_time_s = train_df.lag_time_s.astype(np.int)
    train_df.lag_time_m = train_df.lag_time_m.astype(np.int)
    train_df.lag_time_d = train_df.lag_time_d.astype(np.int)
    del lag_time_col
    gc.collect()

    # 按用户grouping
    # grouping based on user_id to get the data supplu
    print("Grouping users...")
    group = train_df[
        ["user_id", "content_id", "answered_correctly", "prior_question_elapsed_time", "tags", "tags_num", "popularity",
         "difficulty", "lag_time_s", "lag_time_m", "lag_time_d"]] \
        .groupby("user_id") \
        .apply(lambda r: (r.content_id.values, r.answered_correctly.values,
                          r.prior_question_elapsed_time.values, r.tags.values,
                          r.tags_num.values, r.popularity.values, r.difficulty.values,
                          r.lag_time_s.values, r.lag_time_m.values, r.lag_time_d.values))
    # user_id: ID code for the user.
    # content_id: ID code for the user interaction
    # answered_correctly: if the user responded correctly. Read -1 as null, for lectures.
    # prior_question_elapsed_time: The average time in milliseconds it took a user to answer each question
    # task_container_id: Id code for the batch of questions or lectures.
    n_users = train_df.user_id.nunique()
    print("no. of users :", n_users)
    del train_df
    gc.collect()
    print("splitting")
    other, test = train_test_split(group, test_size=0.1)
    train, val = train_test_split(other, test_size=0.2)
    print("train size: ", train.shape, "validation size: ", val.shape, "test size: ", test.shape)

    # if SAVE_DATA_TO_CACHE:
    train.to_pickle(f"{Config.DATA_FILE_PATH}.train")
    val.to_pickle(f"{Config.DATA_FILE_PATH}.valid")
    test.to_pickle(f"{Config.TEST_FILE_PATH}.test")
    print("Data is Ready")

"""
EdNet-KT1
"""
def concat_EdNet_data():
    # EdNet:先对所有用户csv合并
    u_dtypes = {'timestamp': 'int64', 'question_id': 'str',
                'user_answer': 'str', "elapsed_time": "int64",
                'answered_correctly': 'int8', "user_id": "str"}
    flag =True
    for uid in tqdm(os.listdir(Config.TRAIN_FILE)):
        domain = os.path.abspath(Config.TRAIN_FILE)
        u_file = os.path.join(domain, uid)
        u_df = pd.read_csv(u_file, usecols=[
            0, 2, 3, 4], dtype=u_dtypes)
        u_df["user_id"] = uid.replace('.csv', '')
        u_df.to_csv(Config.DATA_FILE_PATH + ".csv", mode='a', header=flag)
        flag = False
        del u_df
        gc.collect()
    print("csv ready")

def get_EdNet_data():
    question_dtypes = {"question_id": "str", "correct_answer": "str", "tags": "str", "tags_num": "int16"}
    u_dtypes = {'timestamp': 'int64', 'question_id': 'str',
              'user_answer': 'str', "elapsed_time": "int64",
              'answered_correctly': 'int8', "user_id": "str"}
    print("loading csv.....")
    question_df = pd.read_csv(Config.QUESTION_FILE, usecols=[
        0, 3, 5, 7], dtype=question_dtypes)
    # Align the name of key column for latter merging
    # question_df = question_df.rename(columns={"question_id": "content_id"})
    # 处理tags格式: str -> list[int]
    question_df.tags = question_df.tags.str.split(';')
    for i in range(len(question_df.tags)):
        question_df.tags.at[i] = np.array(list(map(int, question_df.tags[i])), 'int64')
        question_df.tags.at[i] = np.pad(question_df.tags[i], (0, Config.MAX_CATS_PER - len(question_df.tags[i])),
                                        'constant', constant_values=(0))
    print("question file ready")
    users_df =pd.read_csv(Config.TRAIN_FILE, dtype=u_dtypes,header=0)
    print("user file loaded")
    print("shape of dataframe :", users_df.shape)
    users_df.elapsed_time.fillna(0, inplace=True)
    users_df.elapsed_time /= 3600
    users_df.elapsed_time = users_df.elapsed_time.astype(np.int32)
    # Merging train_df and question_df on question_id
    users_df = users_df.merge(question_df, on='question_id', how="left")  # left outer join to consider part
    del question_df
    gc.collect()
    print("user-question file merged")
    users_df["question_id"] = users_df["question_id"].apply(lambda x : int(x.replace("q","")))
    # response
    users_df["answered_correctly"] = users_df[['user_answer', 'correct_answer']] \
        .apply(lambda x: 1 if x['user_answer'] == x['correct_answer'] else 0, axis=1)
    # 时间排序
    users_df = users_df.sort_values(["timestamp"], ascending=True).reset_index(drop=True)
    print("Computing question difficulty")
    df_difficulty = users_df["answered_correctly"].groupby(users_df["question_id"])
    users_df["popularity"] = df_difficulty.transform('size')
    users_df["difficulty"] = df_difficulty.transform('sum') / users_df["popularity"]
    print("Popularity max", users_df["popularity"].max(), ",Difficulty max", users_df["difficulty"].max())
    del df_difficulty
    gc.collect()

    n_questions = users_df.question_id.nunique()
    print("no. of questions :", n_questions)
    print("shape after exlusion:", users_df.shape)

    print("Calculating lag time")
    time_dict = {}
    lag_time_col = np.zeros(len(users_df), dtype=np.int64)
    for ind, row in enumerate(tqdm(users_df[["user_id", "timestamp", "question_id"]].values)):
        if row[0] in time_dict.keys():
            # if the task_container_id is the same, the lag time is not allowed
            if row[2] == time_dict[row[0]][1]:
                lag_time_col[ind] = time_dict[row[0]][2]
            else:
                timestamp_last = time_dict[row[0]][0]
                lag_time_col[ind] = row[1] - timestamp_last
                time_dict[row[0]] = (row[1], row[2], lag_time_col[ind])
        else:
            time_dict[row[0]] = (row[1], row[2], 0)
            lag_time_col[ind] = 0
        if lag_time_col[ind] < 0:
            raise RuntimeError("Has lag_time smaller than 0.")

    users_df["lag_time_s"] = lag_time_col // 1000
    users_df["lag_time_m"] = lag_time_col // (60 * 1000)
    users_df["lag_time_d"] = lag_time_col // (60 * 1000 * 1440)
    users_df.lag_time_s.clip(lower=0, upper=300, inplace=True)
    users_df.lag_time_m.clip(lower=0, upper=1440, inplace=True)
    users_df.lag_time_d.clip(lower=0, upper=365, inplace=True)
    users_df.lag_time_s = users_df.lag_time_s.astype(np.int)
    users_df.lag_time_m = users_df.lag_time_m.astype(np.int)
    users_df.lag_time_d = users_df.lag_time_d.astype(np.int)
    del lag_time_col
    gc.collect()

    # 按用户grouping
    # grouping based on user_id to get the data supplu
    print("Grouping users...")
    group = users_df[["user_id", "question_id", "answered_correctly", "elapsed_time",
                      "tags", "tags_num", "popularity", "difficulty",
                      "lag_time_s", "lag_time_m", "lag_time_d",]] \
        .groupby("user_id") \
        .apply(lambda r: (r.question_id.values, r.answered_correctly.values,
                          r.elapsed_time.values, r.tags.values,
                          r.tags_num.values, r.popularity.values, r.difficulty.values,
                          r.lag_time_s.values, r.lag_time_m.values, r.lag_time_d.values))
    # user_id: ID code for the user.
    # content_id: ID code for the user interaction
    # answered_correctly: if the user responded correctly. Read -1 as null, for lectures.
    # prior_question_elapsed_time: The average time in milliseconds it took a user to answer each question
    # task_container_id: Id code for the batch of questions or lectures.
    n_users = users_df.user_id.nunique()
    print("no. of users :", n_users)
    del users_df
    gc.collect()
    print("splitting")
    other, test = train_test_split(group, test_size=0.1)
    train, val = train_test_split(other, test_size=0.2)
    print("train size: ", train.shape, "validation size: ", val.shape, "test size: ", test.shape)

    # if SAVE_DATA_TO_CACHE:
    train.to_pickle(f"{Config.DATA_FILE_PATH}.train")
    val.to_pickle(f"{Config.DATA_FILE_PATH}.valid")
    test.to_pickle(f"{Config.TEST_FILE_PATH}.test")
    print("Data is Ready")


"""
ASSIST09
"""
def get_AS09_data(mode):
    question_dtypes = {"problem_id": "int32", "tags": "str", "tags_num": "int16"}
    if mode == "ns":
        correct_t = "float64"
    else:
        correct_t = "int8"
    u_dtypes = {'order_id': 'int64', 'user_id': 'int32',
                'problem_id': 'int32', "correct": correct_t,
                "ms_first_response": "int64",  'overlap_time': 'int64'}
    print("loading csv.....")
    question_df = pd.read_csv(Config.QUESTION_FILE, usecols=[1, 2, 4], dtype=question_dtypes)
    # Align the name of key column for latter merging
    # 处理tags格式: str -> list[int]
    question_df.tags = question_df.tags.str.split(';')
    for i in range(len(question_df.tags)):
        question_df.tags.at[i] = np.array(list(map(int, question_df.tags[i])), 'int32')
        question_df.tags.at[i] = np.pad(question_df.tags[i], (0, Config.MAX_CATS_PER - len(question_df.tags[i])),
                                        'constant', constant_values=(0))
    print("question file ready")
    users_df =pd.read_csv(Config.TRAIN_FILE, dtype=u_dtypes,header=0)
    print("user file loaded")
    print("shape of dataframe :", users_df.shape)
    # AS09中没有elapsed_time 用ms_first_response代替
    users_df.ms_first_response.fillna(0, inplace=True)
    users_df.ms_first_response = abs(users_df.ms_first_response)
    users_df.ms_first_response /= 3600
    users_df.ms_first_response = users_df.ms_first_response.astype(np.int64)
    if mode == "ns":
        users_df.correct = users_df.correct.round().astype(np.int8)
    # Merging train_df and question_df on question_id
    users_df = users_df.merge(question_df, on='problem_id', how="left")  # left outer join to consider part
    del question_df
    gc.collect()
    print("user-question file merged")
    # 时间排序
    users_df = users_df.sort_values(["order_id"], ascending=True).reset_index(drop=True)
    print("Computing question difficulty")
    df_difficulty = users_df["correct"].groupby(users_df["problem_id"])
    users_df["popularity"] = df_difficulty.transform('size')
    users_df["difficulty"] = df_difficulty.transform('sum') / users_df["popularity"]
    print("Popularity max", users_df["popularity"].max(), ",Difficulty max", users_df["difficulty"].max())
    del df_difficulty
    gc.collect()

    n_questions = users_df.problem_id.nunique()
    print("no. of questions :", n_questions)
    print("shape after exlusion:", users_df.shape)
    # AS09中没有timestamp 用orderid代替
    print("Calculating lag time")
    time_dict = {}
    lag_time_col = np.zeros(len(users_df), dtype=np.int64)
    for ind, row in enumerate(tqdm(users_df[["user_id", "order_id", "problem_id"]].values)):
        if row[0] in time_dict.keys():
            # if the task_container_id is the same, the lag time is not allowed
            if row[2] == time_dict[row[0]][1]:
                lag_time_col[ind] = time_dict[row[0]][2]
            else:
                timestamp_last = time_dict[row[0]][0]
                lag_time_col[ind] = row[1] - timestamp_last
                time_dict[row[0]] = (row[1], row[2], lag_time_col[ind])
        else:
            time_dict[row[0]] = (row[1], row[2], 0)
            lag_time_col[ind] = 0
        if lag_time_col[ind] < 0:
            raise RuntimeError("Has lag_time smaller than 0.")
    # AS09中用10 100 1000代替
    users_df["lag_time_s"] = lag_time_col // 10
    users_df["lag_time_m"] = lag_time_col // 100
    users_df["lag_time_d"] = lag_time_col // 1000
    users_df.lag_time_s.clip(lower=0, upper=5000, inplace=True)
    users_df.lag_time_m.clip(lower=0, upper=500, inplace=True)
    users_df.lag_time_d.clip(lower=0, upper=50, inplace=True)
    users_df.lag_time_s = users_df.lag_time_s.astype(np.int)
    users_df.lag_time_m = users_df.lag_time_m.astype(np.int)
    users_df.lag_time_d = users_df.lag_time_d.astype(np.int)
    del lag_time_col
    gc.collect()

    # 按用户grouping
    # grouping based on user_id to get the data supplu
    print("Grouping users...")
    group = users_df[["user_id", "problem_id", "correct", "ms_first_response",
                      "tags", "tags_num", "popularity", "difficulty",
                      "lag_time_s", "lag_time_m", "lag_time_d",]] \
        .groupby("user_id") \
        .apply(lambda r: (r.problem_id.values, r.correct.values,
                          r.ms_first_response.values, r.tags.values,
                          r.tags_num.values, r.popularity.values, r.difficulty.values,
                          r.lag_time_s.values, r.lag_time_m.values, r.lag_time_d.values))
    n_users = users_df.user_id.nunique()
    print("no. of users :", n_users)
    del users_df
    gc.collect()
    print("splitting")
    other, test = train_test_split(group, test_size=0.1)
    train, val = train_test_split(other, test_size=0.2)
    print("train size: ", train.shape, "validation size: ", val.shape, "test size: ", test.shape)

    # if SAVE_DATA_TO_CACHE:
    train.to_pickle(f"{Config.DATA_FILE_PATH}.train")
    val.to_pickle(f"{Config.DATA_FILE_PATH}.valid")
    test.to_pickle(f"{Config.TEST_FILE_PATH}.test")
    print("Data is Ready")


"""
ASSIST12
"""
def get_AS12_data():
    u_dtypes = {'user_id': 'int32', "skill_id": "str", "skill_num": "int16",
                'problem_id': 'int32', "correct": 'int8',
                "timestamp": "int64",  'dwell_time': 'int64'}
    print("loading csv.....")
    # users_df =pd.read_csv(Config.TRAIN_FILE, dtype=u_dtypes,header=0)
    users_df =pd.read_csv(Config.TRAIN_FILE, dtype=u_dtypes,header=0, sep='\t')
    # 处理tags格式: str -> list[int]
    users_df.insert(loc=12, column='skill_num', value=1)
    n_questions = users_df.problem_id.nunique()
    print("no. of questions :", n_questions)
    max_questions = users_df.problem_id.max()
    print("max of questions :", max_questions)
    users_df.skill_id = users_df.skill_id.str.split(';')
    for i in range(len(users_df.skill_id)):
        users_df.skill_id.at[i] = np.array(list(map(int, users_df.skill_id[i])), 'int32')
        users_df.skill_id.at[i] = np.pad(users_df.skill_id[i], (0, Config.MAX_CATS_PER - len(users_df.skill_id[i])),
                                        'constant', constant_values=(0))
    # AS12中dwell_time为end-start
    users_df.dwell_time.fillna(0, inplace=True)
    users_df.dwell_time = abs(users_df.dwell_time)
    users_df.dwell_time /= 3600
    users_df.dwell_time = users_df.dwell_time.astype(np.int64)
    print("user file loaded")
    print("shape of dataframe :", users_df.shape)
    # 时间排序
    users_df = users_df.sort_values(["timestamp"], ascending=True).reset_index(drop=True)
    print("Computing question difficulty")
    df_difficulty = users_df["correct"].groupby(users_df["problem_id"])
    users_df["popularity"] = df_difficulty.transform('size')
    users_df["difficulty"] = df_difficulty.transform('sum') / users_df["popularity"]
    print("Popularity max", users_df["popularity"].max(), ",Difficulty max", users_df["difficulty"].max())
    del df_difficulty
    gc.collect()


    print("shape after exlusion:", users_df.shape)

    print("Calculating lag time")
    time_dict = {}
    lag_time_col = np.zeros(len(users_df), dtype=np.int64)
    for ind, row in enumerate(tqdm(users_df[["user_id", "timestamp", "problem_id"]].values)):
        if row[0] in time_dict.keys():
            # if the task_container_id is the same, the lag time is not allowed
            if row[2] == time_dict[row[0]][1]:
                lag_time_col[ind] = time_dict[row[0]][2]
            else:
                timestamp_last = time_dict[row[0]][0]
                lag_time_col[ind] = row[1] - timestamp_last
                time_dict[row[0]] = (row[1], row[2], lag_time_col[ind])
        else:
            time_dict[row[0]] = (row[1], row[2], 0)
            lag_time_col[ind] = 0
        if lag_time_col[ind] < 0:
            raise RuntimeError("Has lag_time smaller than 0.")

    users_df["lag_time_s"] = lag_time_col // 1000
    users_df["lag_time_m"] = lag_time_col // (60 * 1000)
    users_df["lag_time_d"] = lag_time_col // (60 * 1000 * 1440)
    users_df.lag_time_s.clip(lower=0, upper=300, inplace=True)
    users_df.lag_time_m.clip(lower=0, upper=1440, inplace=True)
    users_df.lag_time_d.clip(lower=0, upper=365, inplace=True)
    users_df.lag_time_s = users_df.lag_time_s.astype(np.int)
    users_df.lag_time_m = users_df.lag_time_m.astype(np.int)
    users_df.lag_time_d = users_df.lag_time_d.astype(np.int)
    del lag_time_col
    gc.collect()

    # 按用户grouping
    # grouping based on user_id to get the data supplu
    print("Grouping users...")
    group = users_df[["user_id", "problem_id", "correct", "dwell_time",
                      "skill_id", "skill_num", "popularity", "difficulty",
                      "lag_time_s", "lag_time_m", "lag_time_d",]] \
        .groupby("user_id") \
        .apply(lambda r: (r.problem_id.values, r.correct.values,
                          r.dwell_time.values, r.skill_id.values, r.skill_num.values,
                          r.popularity.values, r.difficulty.values,
                          r.lag_time_s.values, r.lag_time_m.values, r.lag_time_d.values))
    n_users = users_df.user_id.nunique()
    print("no. of users :", n_users)
    del users_df
    gc.collect()
    print("splitting")
    other, test = train_test_split(group, test_size=0.1)
    train, val = train_test_split(other, test_size=0.2)
    print("train size: ", train.shape, "validation size: ", val.shape, "test size: ", test.shape)

    # if SAVE_DATA_TO_CACHE:
    train.to_pickle(f"{Config.DATA_FILE_PATH}.train")
    val.to_pickle(f"{Config.DATA_FILE_PATH}.valid")
    test.to_pickle(f"{Config.TEST_FILE_PATH}.test")
    print("Data is Ready")

def get_AS12_simple_data():
    u_dtypes = {'problem_log_id':'int64','skill':'str','problem_id': 'int32',
                'user_id': 'int32', "start_time": "str",  'end_time': 'str',
                'correct':'str',"skill_id": "str","overlap_time": 'int64'}
    print("loading csv.....")
    users_df =pd.read_csv(Config.TRAIN_FILE, dtype=u_dtypes,header=0)
    users_df[u_dtypes].to_csv('../data/AS12/AS12.csv')
    users_df = users_df[u_dtypes]
    users_df.drop_duplicates(subset='problem_id', keep='first').to_csv('../data/AS12/AS12_problem.csv')

def get_AS12_corrected_data():
    u_dtypes = {'problem_log_id':'int64','skill':'str','problem_id': 'int32',
                'user_id': 'int32', "start_time": "str",  'end_time': 'str',
                'correct':'float64',"skill_id": "str","overlap_time": 'int64'}
    print("loading csv.....")
    users_df =pd.read_csv(Config.TRAIN_FILE, dtype=u_dtypes,header=0)
    # 处理tags格式: str -> list[int]
    # 补充skill_num 和 skill_id空值
    users_df['correct'].astype(np.int64)
    users_df.insert(loc=10, column='skill_num', value=1)
    users_df['skill_id'] = users_df['skill_id'].fillna('0')
    n_questions = users_df.problem_id.nunique()
    print("no. of questions :", n_questions)
    print("max. of questions :", users_df.problem_id.max())
    n_skills = users_df.skill_id.nunique()
    print("no. of skills :", n_skills)
    for i in range(len(users_df.skill_id)):
        users_df.skill_id.at[i] = np.array(list(map(int, users_df.skill_id[i])), 'int32')
        users_df.skill_id.at[i] = np.pad(users_df.skill_id[i], (0, Config.MAX_CATS_PER - len(users_df.skill_id[i])),
                                        'constant', constant_values=(0))
    # users_df.skill_id = users_df.skill_id.str.split(';')
    # for i in range(len(users_df.skill_id)):
    #     users_df.skill_id.at[i] = np.array(list(map(int, users_df.skill_id[i])), 'int32')
    #     users_df.skill_id.at[i] = np.pad(users_df.skill_id[i], (0, Config.MAX_CATS_PER - len(users_df.skill_id[i])),
    #                                     'constant', constant_values=(0))
    # AS12中elapsed_time为end-start
    users_df["elapsed_time"] = pd.to_datetime(users_df["end_time"]) - pd.to_datetime(users_df["start_time"])
    users_df["elapsed_time"] = (
        users_df["elapsed_time"].apply(lambda x: x.total_seconds()).astype(np.int64)
    )
    # AS12中没有timestamp 用start_time代替
    users_df["timestamp"] = pd.to_datetime(users_df["start_time"])
    users_df["timestamp"] = users_df["timestamp"] - users_df["timestamp"].min()
    users_df["timestamp"] = (
        users_df["timestamp"].apply(lambda x: x.total_seconds()).astype(np.int64)
    )
    # users_df.dwell_time.fillna(0, inplace=True)
    # users_df.dwell_time = abs(users_df.dwell_time)
    # users_df.dwell_time /= 3600
    # users_df.dwell_time = users_df.dwell_time.astype(np.int64)

    print("user file loaded")
    print("shape of dataframe :", users_df.shape)
    # 时间排序
    users_df = users_df.sort_values(["timestamp"], ascending=True).reset_index(drop=True)
    print("Computing question difficulty")
    df_difficulty = users_df["correct"].groupby(users_df["problem_id"])
    users_df["popularity"] = df_difficulty.transform('size')
    users_df["difficulty"] = df_difficulty.transform('sum') / users_df["popularity"]
    print("Popularity max", users_df["popularity"].max(), ",Difficulty max", users_df["difficulty"].max())
    del df_difficulty
    gc.collect()
    print("shape after exlusion:", users_df.shape)

    print("Calculating lag time")
    time_dict = {}
    lag_time_col = np.zeros(len(users_df), dtype=np.int64)
    for ind, row in enumerate(tqdm(users_df[["user_id", "timestamp", "problem_id"]].values)):
        if row[0] in time_dict.keys():
            # if the task_container_id is the same, the lag time is not allowed
            if row[2] == time_dict[row[0]][1]:
                lag_time_col[ind] = time_dict[row[0]][2]
            else:
                timestamp_last = time_dict[row[0]][0]
                lag_time_col[ind] = row[1] - timestamp_last
                time_dict[row[0]] = (row[1], row[2], lag_time_col[ind])
        else:
            time_dict[row[0]] = (row[1], row[2], 0)
            lag_time_col[ind] = 0
        if lag_time_col[ind] < 0:
            raise RuntimeError("Has lag_time smaller than 0.")

    users_df["lag_time_s"] = lag_time_col // 1000
    users_df["lag_time_m"] = lag_time_col // (60 * 1000)
    users_df["lag_time_d"] = lag_time_col // (60 * 1000 * 1440)
    users_df.lag_time_s.clip(lower=0, upper=300, inplace=True)
    users_df.lag_time_m.clip(lower=0, upper=1440, inplace=True)
    users_df.lag_time_d.clip(lower=0, upper=365, inplace=True)
    users_df.lag_time_s = users_df.lag_time_s.astype(np.int)
    users_df.lag_time_m = users_df.lag_time_m.astype(np.int)
    users_df.lag_time_d = users_df.lag_time_d.astype(np.int)
    del lag_time_col
    gc.collect()

    # 按用户grouping
    # grouping based on user_id to get the data supplu
    print("Grouping users...")
    group = users_df[["user_id", "problem_id", "correct", "elapsed_time",
                      "skill_id", "skill_num", "popularity", "difficulty",
                      "lag_time_s", "lag_time_m", "lag_time_d",]] \
        .groupby("user_id") \
        .apply(lambda r: (r.problem_id.values, r.correct.values,
                          r.elapsed_time.values, r.skill_id.values, r.skill_num.values,
                          r.popularity.values, r.difficulty.values,
                          r.lag_time_s.values, r.lag_time_m.values, r.lag_time_d.values))
    n_users = users_df.user_id.nunique()
    print("no. of users :", n_users)
    del users_df
    gc.collect()
    print("splitting")
    other, test = train_test_split(group, test_size=0.1)
    train, val = train_test_split(other, test_size=0.2)
    print("train size: ", train.shape, "validation size: ", val.shape, "test size: ", test.shape)

    # if SAVE_DATA_TO_CACHE:
    train.to_pickle(f"{Config.DATA_FILE_PATH}.train")
    val.to_pickle(f"{Config.DATA_FILE_PATH}.valid")
    test.to_pickle(f"{Config.TEST_FILE_PATH}.test")
    print("Data is Ready")

if __name__ == "__main__":
    start_time = time()
    # concat_EdNet_data()
    # get_EdNet_data()

    # get_RAIEd_data()

    # get_AS09_data("ns")

    get_AS12_corrected_data()
    print("TOTAL TIME: ", time() - start_time)
