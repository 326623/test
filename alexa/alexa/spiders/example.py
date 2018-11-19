# -*- coding: utf-8 -*-
import scrapy
from scrapy.linkextractors import LinkExtractor


# 一个域名大概20个url
# 共5个深度，每个深度爬4，5个URL

"""
Every CrawlSpider takes input from a list of urls,
every spider will maintain a special data structure
to keep track of metadata of already crawled urls.

And a set of rules to decide whether to keep the
result. After crawling all urls, write to disk based
on some other rules.

The first set of rules is mostly whether the url fits
in the original domain name.

The second set of rules will try to strike a balance of
different depths of url.

"""

class AlexaSpider(scrapy.spiders.CrawlSpider):
    name = 'example'
    allowed_domains = ['twitter.com']
    start_urls = ['http://twitter.com/']

    rules = (scrapy.spiders.Rule(LinkExtractor(), callback='parse_url', follow=False),)

    def parse_url(self, response):
        return response.url
