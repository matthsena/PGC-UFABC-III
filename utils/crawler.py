import os
import re
import time
from PIL import Image
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import sys
import urllib.request

dataset = sys.argv[1]
class ImageDownloader:
    
    def __init__(self, url, output_directory):
        self.url = url
        self.output_directory = output_directory

    def start_download(self):
        try:
            options = self._get_chrome_options()
            webdriver_service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=webdriver_service, options=options)
            driver.get(self.url)

            WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "img")))

            img_elements = driver.find_elements(By.TAG_NAME, "img")
            img_links = [img.get_attribute('src') for img in img_elements]

            self._create_directory_if_not_exists()

            for i, img_link in enumerate(img_links):
                self._process_image_link(img_link)

            driver.quit()
        except Exception as e:
            print(f"[CRAWLER] Erro ao iniciar o download da imagem: {e}")

    def _get_chrome_options(self):
        options = Options()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--headless")
        return options

    def _create_directory_if_not_exists(self):
        if not os.path.isdir(self.output_directory):
            os.makedirs(self.output_directory)

    def _process_image_link(self, img_link):
        pattern = r'/(\d+)px-'
        matches = re.findall(pattern, img_link)

        if len(matches):
            if int(matches[0]) >= 100:
                self._download_image(img_link)

    def _download_image(self, img_link):
        url_img = self._replace_px_value(img_link)
        filename = url_img.split('/')[-1]

        if not filename.split('.')[-1] == 'svg':
            urllib.request.urlretrieve(url_img, os.path.join(self.output_directory, filename))
            print(f'[CRAWLER] Imagem baixada: {filename}')

            self._check_and_convert_image(os.path.join(self.output_directory, filename))

    def _replace_px_value(self, string):
        modified_string = re.sub(r'\b\d+px', '2240px', string)
        return modified_string

    def _is_gif(self, filename):
        return filename.split('.')[-1] == 'gif'

    def _gif_to_png(self, filename):
        with Image.open(filename) as im:
            im.seek(im.n_frames - 1)
            png_filename = filename.replace('.gif', '.png')
            im.save(png_filename, 'png')
            return png_filename

    def _check_and_convert_image(self, filename):
        if self._is_gif(filename):
            png_filename = self._gif_to_png(filename)
            return png_filename
        else:
            return filename

def main():
    try:
        df = pd.read_csv(f'utils/datasets/{dataset}.csv')

        for _, row in df.iterrows():
            img_path = f"images/{dataset}/{row['title']}/{row['lang']}"

            if row['url'] != '':
                downloader = ImageDownloader(row['url'], img_path)
                downloader.start_download()
            else:
                if not os.path.isdir(img_path):
                    os.makedirs(img_path)
    except Exception as e:
        print(f"[CRAWLER] Erro no módulo do Crawler: {e}")

if __name__ == "__main__":
    main()
