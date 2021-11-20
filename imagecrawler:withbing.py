import pandas as pd
from icrawler.builtin import GoogleImageCrawler
from icrawler.builtin import BingImageCrawler
from pathlib import Path

mitFilter=True
#Set the filter to creative vommons license and set if th eimage is either photo, face, clipart, linedrawing, or animated
filters = dict(
    type='photo',
    # license='commercial,modify')
    license='noncommercial')

howmany= 10
names=pd.read_csv('Top 1000 Actors and Actresses.csv', encoding = "ISO-8859-1"),

subset=names[0]['Name']
   
for keyword in subset:
    crawler = BingImageCrawler(
        parser_threads=5,
        downloader_threads=5,
        storage={'root_dir': 'Celebs/{}'.format(keyword)}
    )    
    if mitFilter==True:
        crawler.crawl(keyword=keyword, filters=filters, max_num=howmany, min_size=(500, 500))
    else:
        crawler.crawl(keyword=keyword, max_num=howmany, min_size=(500, 500))