import numpy as np
import pandas as pd
import math
from hashlib import sha256


class ScoreCalculator:
    def __init__(self, N_LANGS=8):
        self.N_LANGS = N_LANGS

    def __parabola_score(self, x):
        if x <= (self.N_LANGS // 2):
            return np.exp(-(x - 1) / math.pi)
        return np.exp((x) / math.pi) / np.exp(self.N_LANGS / math.pi)

    def __decay_score(self, x):
        return np.exp(-(x - 1) / math.pi)

    def __growth_score(self, x):
        return np.exp((x) / math.pi) / np.exp(self.N_LANGS / math.pi)

    def __simple_score(self, x):
        return 2 if x == 1 else 1

    def calculate_score(self, df, article_name):
        df['id'] = df['original_photo'].apply(
            lambda x: sha256(x.encode('utf-8')).hexdigest())

        filtered_df = df[df['distance'] <= 0.25]

        filtered_no_eq_df = df[~df['id'].isin(
            filtered_df['id']) & df['distance'] > 0.25]
        filtered_no_eq_df = filtered_no_eq_df.drop_duplicates(subset=['id'])

        uniques_df = pd.DataFrame({
            'original_photo': filtered_no_eq_df['original_photo'],
            'languages':  filtered_no_eq_df['original'].apply(lambda x: [x]),
            'num_languages': 1
        }).reset_index()

        similar_df = (filtered_df.groupby('original_photo')
                      .apply(lambda x: pd.Series({
                          'languages': list(set(x['original']).union(set(x['compare']))),
                          'num_languages': len(list(set(x['original']).union(set(x['compare'])))),
                          'id': x['id']
                      }))
                      .reset_index())

        grouped_df = pd.concat([similar_df, uniques_df])
        sorted_df = grouped_df.sort_values(by='num_languages', ascending=False)

        sorted_df['decay'] = sorted_df['num_languages'].apply(
            lambda x: self.__decay_score(x))
        sorted_df['growth'] = sorted_df['num_languages'].apply(
            lambda x: self.__growth_score(x))
        sorted_df['parabola'] = sorted_df['num_languages'].apply(
            lambda x: self.__parabola_score(x))
        sorted_df['simple'] = sorted_df['num_languages'].apply(
            lambda x: self.__simple_score(x))

        langs_to_check = ['pt', 'en', 'es', 'de', 'it', 'ru', 'zh', 'fr']

        dict_scores_decay = {}
        dict_scores_growth = {}
        dict_scores_parabola = {}
        dict_scores_simple = {}

        for lang in langs_to_check:
            lang_df = sorted_df[sorted_df['languages'].apply(
                lambda x: lang in x)]

            dict_scores_decay[lang] = lang_df['decay'].sum()
            dict_scores_growth[lang] = lang_df['growth'].sum()
            dict_scores_parabola[lang] = lang_df['parabola'].sum()
            dict_scores_simple[lang] = lang_df['simple'].sum()

        scores = {
            'decay': dict_scores_decay,
            'growth': dict_scores_growth,
            'parabola': dict_scores_parabola,
            'simple': dict_scores_simple,
        }
        scores_list = [{'type': score_type, 'article': article_name, **scores}
                       for score_type, scores in scores.items()]
        return scores_list
