import pandas as pd
import model
import time
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score

dataset = pd.read_excel('dataset_comments_100_test.xlsx')
n = len(dataset['MessageText'])

classes = ['B', 'N', 'G']

results = pd.DataFrame({'Name': [],
                        'Accuracy': [],
                        'Precision B': [],
                        'Precision N': [],
                        'Precision G': [],
                        'Recall B': [],
                        'Recall N': [],
                        'Recall G': [],
                        })


def metrix(data, name):
    cm = confusion_matrix(dataset['Class'], data, labels=classes)
    print(f'{name}\n{cm}')
    res = [0 for _ in range(8)]
    res[0] = name
    accuracy = accuracy_score(dataset['Class'], data)
    res[1] = accuracy
    precision = precision_score(dataset['Class'], data, average=None)
    res[2] = precision[0]
    res[3] = precision[2]
    res[4] = precision[1]
    recall = recall_score(dataset['Class'], data, average=None)
    res[5] = recall[0]
    res[6] = recall[2]
    res[7] = recall[1]
    results.loc[len(results)] = res
    file.write(f'{";".join(map(str, res))}\n')
    # print(res)



file = open('Запаска2.txt', 'w')
file.write('Name;Accuracy;Precision B;Precision N;Precision G;Recall B;Recall N;Recall G\n')



# Перебор параметров
for rt in ['score-label', 'label']: #2
    for pas_thresh in range(0, 110, 20): #5
        pas_thresh /= 100
        for coef in range(0, 310, 60): #5
            coef /= 100
            for stb in range(0, 110, 20): #5
                stb /= 100
                for name_thresh in range(0, 110, 20): #5
                    name_thresh /= 100
                    start_time = time.time()
                    sentiments = [model.get_sentiment(text, return_type=rt, passing_threshold=pas_thresh, coefficient=coef,
                                                      start_boost=stb, name_thresh=name_thresh)
                                  for text in dataset['MessageText']]
                    metrix(sentiments, f'{rt}, {pas_thresh}, {coef}, {stb}, {name_thresh}')
                    print(f'Время: {time.time() - start_time}')
file.close()
results.to_excel('Результаты тестов2.xlsx')