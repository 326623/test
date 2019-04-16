# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
import scrapy
from scrapy.pipelines.files import FilesPipeline
from scrapy.exceptions import DropItem
import os

class DownloadPipeline(FilesPipeline):
    def file_path(self, request, response=None, info=None):
        # original_path = super(DownloadPipeline, self).file_path(
        #     request, response=None, info=None)
        print(request.meta.get('filename', ''))
        return os.path.join('full', request.meta.get('filename', ''))

    def get_media_requests(self, item, info):
        # file_url = item['file_urls']
        # meta = {'filenames': item['names']}
        for file_url, name in zip(item['file_urls'], item['names']):
            yield scrapy.Request(url=file_url, meta={'filename': name})

    # def item_completed(self, results, item, info):
    #     # file_paths = [x['paths'] for ok, x in results if ok]
    #     # print(file_paths)
    #     # if not file_paths:
    #     #     raise DropItem('item contains no files')
    #     # else:
    #     #     for ok, x in results:
    #     #         if ok:
    #     #             print(x)
    #     for ok, x in results:
    #         print(ok, x)
    #     return item
