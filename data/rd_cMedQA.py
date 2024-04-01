import pandas as pd
import os
import json
from tqdm import tqdm

folder_path = "./"

questions_df = pd.read_csv(os.path.join(folder_path, "question.csv"), encoding='utf-8')
answers_df = pd.read_csv(os.path.join(folder_path, "answer.csv"))
merged = pd.merge(questions_df, answers_df, on='question_id')

data = []

for index, row in tqdm(merged.iterrows()):
    q_id = row['question_id']
    q_content = row['content_x']

    a_id = row['ans_id']
    a_content = row['content_y']

    info = {
        "instruction": str(q_content),
        "input": "",
        "output": str(a_content)
    }
    data.append(info)

with open('cMedQA.json', 'w+', encoding='utf-8') as f:
    json.dump(data, f)

