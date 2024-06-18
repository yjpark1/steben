"""
get SteinLib datasets
https://steinlib.zib.de/steinlib.php
"""
import abc
import base64
import logging
import os
import time
import typing
import glob
from collections import namedtuple

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup


logging.basicConfig(
    format='%(asctime)s %(levelname)s:%(message)s',
    level=logging.DEBUG,
    datefmt='%m/%d/%Y %I:%M:%S %p',
)

logger = logging.getLogger("datacrawler.dev")


# __dataset_names = ['SteinLib', 'Vienna', 'Copenhagen14', 'PUCN', 'GAPS']
# __datasets = {k: [] for k in __dataset_names}
# files = glob.iglob('data/**/raws/*')

# for filepath in files:
#     filepath_ = filepath.split('/')
#     _, dataset_name, _, filename = filepath_
#     if dataset_name in __dataset_names:
#         __datasets[dataset_name].append(filepath)
#     else:
#         raise NotImplementedError(f"Unsupported datasets: {dataset_name}")


class BenchmarkDataset(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        self.path_tmp = '../tmp/'
        os.makedirs(self.path_tmp, exist_ok=True)
        self.init_dataset()
        self.init_request_header()
    
    @abc.abstractmethod
    def init_dataset(self):
        pass
    
    @abc.abstractmethod
    def get_dataset_urls(self) -> typing.List[typing.Dict]:
        pass
    
    def init_request_header(self):
        self.request_header = {
            # 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:10.0) Gecko/20100101 Firefox/33.0'
        }
    
    def request_url(self, url):
        path_for_url = self.path_tmp + self._encode_url(url)
        if os.path.exists(path_for_url):
            # load file
            with open(path_for_url, 'rt', encoding='utf-8') as f:
                content = f.read()
                logger.info('read html from tmp file...')
        else:
            # request
            content = self._request_url(url)
            logger.info('get html by request')
        return content
        
    def _request_url(self, url):
        try:
            resp = requests.get(url)
        except requests.exceptions.RequestException as err:
            raise err
        else:
            if resp.ok:
                url_encode = self._encode_url(url)
                with open(self.path_tmp + url_encode, 'w') as f:
                    f.write(resp.text)
                return resp.text
            else:
                return None
            
    def _make_soup(self, html):
        soup = BeautifulSoup(html, 'lxml')
        return soup
    
    def _encode_url(self, url):
        url_encode = base64.b64encode(url.encode('ascii'))
        return url_encode.decode()
    
    def download_datasets(self, urls_dataset: typing.List[typing.Dict]):
        os.makedirs(f'data/{self.name}/raws', exist_ok=True)
        for i, url in enumerate(urls_dataset):
            self._download_single_dataset(url['datalinks'])
        logger.info('all datasets are downloaded.')
        
    def _download_single_dataset(self, urls: typing.List):
        for url in urls:
            filename = url.split('/')[-1]
            filepath = f'data/{self.name}/raws/{filename}'
            if not os.path.exists(filepath):
                r = requests.get(url, headers=self.request_header)
                with open(filepath, 'wb') as file:
                    file.write(r.content)
                time.sleep(3)
            else:
                logger.info('datasets already exists.')


class SteinLib(BenchmarkDataset):
    """_summary_

    Args:
        BenchmarkDataset (_type_): _description_
        
    sample url for download: https://steinlib.zib.de/download/B.tgz
    """
    def __init__(self) -> None:
        super().__init__()
        self.name = 'SteinLib'
        
    def init_dataset(self):
        self.root_url = 'https://steinlib.zib.de/'
        self.main_url = self.root_url + 'download.php'
        
    def get_datasets(self):
        urls_dataset = self.get_dataset_urls()
        self.download_datasets(urls_dataset)
        
    def get_dataset_urls(self):
        filename = f'urls_{self.name}.yaml'
        filepath = self.path_tmp + filename
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                dt_urls = yaml.load(f, Loader=yaml.FullLoader)
            logger.info('read urls for datasets from tmp file...')
        else:
            dt_urls = self._get_dataset_urls()
            # try:
            #     with open(filepath, 'w') as f:
            #         for i, line in enumerate(dt_urls):
            #             yaml.dump(line, f)
            # except Exception as e:
            #     os.remove(filepath)
            #     pass
            #     # raise e
        return dt_urls
    
    def _get_dataset_urls(self):
        content = self.request_url(self.main_url)
        soup = self._make_soup(content)
        
        table = soup.select_one('table')
        rows = table.find_all('tr')
        
        datasets = []
        for i, row in enumerate(rows):
            if i == 0:
                Contents = self._make_header_for_row(row)
            else:
                row_contents = [x.contents for x in row.find_all('td')]
                contents = Contents(*row_contents)
                
                name = contents.Name[0]
                dataname = name.contents[0]
                url_metapage = self.root_url + name['href']
                url_datalink = self._get_downloadlinks(contents)
                
                data = {
                    'name': dataname, 
                    'metapage': url_metapage,
                    'datalinks': url_datalink,
                }
                datasets.append(data)
        return datasets
    
    def _make_header_for_row(self, row):
        header = [x.contents[0] for x in row.find_all('th')]
        header = [x.replace(' ', '_') for x in header]
        return namedtuple('contents', header)
    
    def _get_downloadlinks(self, contents):
        url_datalink = []
        for c in contents.Files:
            try:
                url_ = c.get('href')
            except AttributeError as e:
                pass
            else:
                if url_ is not None:
                    url = self.root_url + url_
                    url_datalink.append(url)
        return url_datalink
                
    def download_datasets(self, urls_dataset: typing.List[typing.Dict]):
        os.makedirs(f'data/{self.name}/raws', exist_ok=True)
        for i, url in enumerate(urls_dataset):
            self._download_single_dataset(url['datalinks'])
        logger.info('all datasets are downloaded.')
        
    def _download_single_dataset(self, urls: typing.List):
        for url in urls:
            filename = url.split('/')[-1]
            filepath = f'data/{self.name}/raws/{filename}'
            if not os.path.exists(filepath):
                r = requests.get(url)
                with open(filepath, 'wb') as file:
                    file.write(r.content)
                time.sleep(3)
            else:
                logger.info('datasets already exists.')
    
    
class Vienna(BenchmarkDataset):
    """_summary_

    Args:
        BenchmarkDataset (_type_): _description_
        
    """
    def __init__(self) -> None:
        super().__init__()
        self.name = 'Vienna'
        
    def init_dataset(self):
        self.root_url = 'https://dimacs11.zib.de'
        self.main_url = self.root_url + '/instances/vienna.zip'
        
    def get_datasets(self):
        url_dataset = self.get_dataset_urls()
        self.download_datasets(url_dataset)
        
    def get_dataset_urls(self):
        dt_urls = []
        dt_urls.append(
            {'datalinks': [self.main_url]}
        )
        return dt_urls
    
    
class Copenhagen14(BenchmarkDataset):
    """_summary_

    Args:
        BenchmarkDataset (_type_): _description_
    https://dimacs11.zib.de/instances/Copenhagen14.zip
    """
    def __init__(self) -> None:
        super().__init__()
        self.name = 'Copenhagen14'
        
    def init_dataset(self):
        self.root_url = 'https://dimacs11.zib.de'
        self.main_url = self.root_url + '/instances/Copenhagen14.zip'
        
    def get_datasets(self):
        url_dataset = self.get_dataset_urls()
        self.download_datasets(url_dataset)
        
    def get_dataset_urls(self):
        dt_urls = []
        dt_urls.append(
            {'datalinks': [self.main_url]}
        )
        return dt_urls
    

class PUCN(BenchmarkDataset):
    """_summary_

    Args:
        BenchmarkDataset (_type_): _description_
    https://dimacs11.zib.de/instances/SPG-PUCN.zip
    """
    def __init__(self) -> None:
        super().__init__()
        self.name = 'PUCN'
        
    def init_dataset(self):
        self.root_url = 'https://dimacs11.zib.de'
        self.main_url = self.root_url + '/instances/SPG-PUCN.zip'
        
    def get_datasets(self):
        url_dataset = self.get_dataset_urls()
        self.download_datasets(url_dataset)
        
    def get_dataset_urls(self):
        dt_urls = []
        dt_urls.append(
            {'datalinks': [self.main_url]}
        )
        return dt_urls
    

class GAPS(BenchmarkDataset):
    """_summary_

    Args:
        BenchmarkDataset (_type_): _description_
    https://dimacs11.zib.de/instances/SPG-GAPS.zip
    """
    def __init__(self) -> None:
        super().__init__()
        self.name = 'GAPS'
        
    def init_dataset(self):
        self.root_url = 'https://dimacs11.zib.de'
        self.main_url = self.root_url + '/instances/SPG-GAPS.zip'
        
    def get_datasets(self):
        url_dataset = self.get_dataset_urls()
        self.download_datasets(url_dataset)
        
    def get_dataset_urls(self):
        dt_urls = []
        dt_urls.append(
            {'datalinks': [self.main_url]}
        )
        return dt_urls


if __name__ == "__main__":
    SteinLib().get_datasets()
    Vienna().get_datasets()
    Copenhagen14().get_datasets()
    PUCN().get_datasets()
    GAPS().get_datasets()