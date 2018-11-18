# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor


# 一个域名大概20个url
# 共5个深度，每个深度爬4，5个URL
class AlexaSpider(scrapy.spiders.CrawlSpider):
    name = 'example'
    allowed_domains = ['twitter.com']
    start_urls = ['http://twitter.com/']

    rules = (scrapy.spiders.Rule(LinkExtractor(), callback='parse_url', follow=False),)

    def parse_url(self, response):
        return response.url
