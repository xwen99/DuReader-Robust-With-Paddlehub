import json
import numpy as np
from tqdm import tqdm

def ensemble_example(answers, n_models=None):
    if n_models is None:
        n_models = len(answers)
    answer_dict = dict()
    for nbest_predictions in answers:
        for prediction in nbest_predictions:
            score_list = answer_dict.setdefault(prediction['text'], [])
            score_list.append(prediction['probability'])

    ensemble_nbest_predictions = []
    for answer, scores in answer_dict.items():
        prediction = dict()
        prediction['text'] = answer
        prediction['probability'] = np.sum(scores) / n_models
        ensemble_nbest_predictions.append(prediction)

    ensemble_nbest_predictions = \
        sorted(ensemble_nbest_predictions, key=lambda item: item['probability'], reverse=True)
    return ensemble_nbest_predictions

if __name__ == "__main__":
    pred = {}
    nbest0 = json.load(open('./dureader+dev+ernie_nbest_pred.json', 'r'))
    nbest1 = json.load(open('./dureader+dev+roberta_nbest_pred.json', 'r'))
    nbests = [nbest0, nbest1]
    n_models = len(nbests)
    qids = list(nbests[0].keys())
    for qid in qids:
        ensemble_nbest = ensemble_example([nbest[qid] for nbest in nbests], n_models=n_models)
        pred[qid] = ensemble_nbest[0]['text']
    
    json.dump(pred, open('ensembled_pred.json', 'w'), ensure_ascii=False)