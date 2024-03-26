import os
import sys
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
from itertools import combinations, product
from models.vgg19 import FeatureExtractor
from utils.scores import ScoreCalculator
import matplotlib.pyplot as plt
import concurrent.futures


class ImgFeatureExtractor:
    def __init__(self, dir):
        self.dir = dir
        self.extractor = FeatureExtractor()
        self.folders = [f for f in os.listdir(
            self.dir) if os.path.isdir(os.path.join(self.dir, f))]

    def get_img_files(self, folder, path):
        folder_path = os.path.join(folder, path)
        file_list = [file for file in os.listdir(
            folder_path) if os.path.isfile(os.path.join(folder_path, file))]
        return (path, sorted(file_list))

    def extract_features(self, base_folder, img_files):
        features = {}
        for lang, photos in img_files:
            for photo in photos:
                img_path = os.path.join(lang, photo)
                img = os.path.join(base_folder, img_path)
                features[img_path] = self.extractor.extract_features(img)
        return features

    def get_comparison_list(self, img_files):
        compare_list = []
        seen = set()
        for (lang1, imgs1), (lang2, imgs2) in combinations(img_files, 2):
            for img1, img2 in product(imgs1, imgs2):
                items = tuple(sorted([lang1, lang2, img1, img2]))
                if items not in seen:
                    seen.add(items)
                    compare_list.append(((lang1, lang2), (img1, img2)))
        return compare_list

    def get_results(self, base_folder, compare_list, features):
        results = []
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.extractor.compare_features, features[os.path.join(lang1, img1)], features[os.path.join(
                lang2, img2)]): (lang1, lang2, img1, img2) for (lang1, lang2), (img1, img2) in compare_list}
            for future in concurrent.futures.as_completed(futures):
                lang1, lang2, img1, img2 = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print('[APP] (%r) Erro no app: %s' % (lang1, exc))
                else:
                    results.append({
                        'article': base_folder,
                        'original': lang1,
                        'compare': lang2,
                        'original_photo': img1,
                        'compare_photo': img2,
                        'distance': result
                    })
        return results

    def plot_bar(self, dfs, titles, article):
        _, axes = plt.subplots(nrows=1, ncols=len(dfs), figsize=(15, 5))
        for df, ax, title in zip(dfs, axes, titles):
            df.plot(kind='bar', ax=ax, legend=True)
            mean = df.mean().mean()
            median = df.stack().median()
            ax.axhline(mean, color='red', linestyle='--', label='Mean')
            ax.axhline(median, color='blue', linestyle='-.', label='Median')
            ax.set_ylabel('Score')
            ax.set_xlabel('Langs')
            ax.set_title(title)
            ax.grid(True)
            ax.legend()
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.tight_layout()
        plt.savefig(f'plots/{article}.png')

    def generate_plots(self, df):
        articles = df.article.unique()
        articles = np.array(articles)
        articles = np.sort(articles)
        for article in articles:
            df_article = df[df['article'] == article]
            df_simple = df_article[df_article['type'] == 'simple']
            df_simple = df_simple.drop(columns=['type', 'article'])
            df_decay = df_article[df_article['type'] == 'decay']
            df_decay = df_decay.drop(columns=['type', 'article'])
            dfgrowth = df_article[df_article['type'] == 'growth']
            dfgrowth = dfgrowth.drop(columns=['type', 'article'])
            df_parabola = df_article[df_article['type'] == 'parabola']
            df_parabola = df_parabola.drop(columns=['type', 'article'])
            self.plot_bar([df_simple, df_decay, dfgrowth, df_parabola], [
                     'simple', 'decaimento', 'growth', 'parabola'], article)
            
    def run(self):
        start = time.time()
        final_df = pd.DataFrame()
        langs = ['pt', 'en', 'es', 'de', 'it', 'ru', 'zh', 'fr']
        for folder in self.folders:
            base_path = f'{self.dir}/{folder}/'
            img_files = list(map(lambda path: self.get_img_files(
                base_path, path), langs))
            features = self.extract_features(base_path, img_files)
            compare_list = self.get_comparison_list(img_files)
            results = self.get_results(folder, compare_list, features)
            df_result = pd.DataFrame(results)
            score = ScoreCalculator()
            score_z = score.calculate_score(df_result, folder)
            df = pd.DataFrame(score_z)
            final_df = pd.concat([final_df, df])
        self.generate_plots(final_df)
        end = time.time()
        elapsed = (end - start) // 60
        print(
            f"[APP]: {elapsed} minutos e {(end - start) % 60:.2f} segundos")


if __name__ == "__main__":
    dataset = sys.argv[1]
    data_dir = f'images/{dataset}'
    extractor = ImgFeatureExtractor(data_dir)
    extractor.run()
