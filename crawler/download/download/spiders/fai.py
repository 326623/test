# -*- coding: utf-8 -*-
import scrapy
import urllib.parse
from download.items import Slide

class FaiSpider(scrapy.Spider):
    name = 'fai'
    allowed_domains = ['fai.cs.uni-saarland.de']
    start_urls = ['http://fai.cs.uni-saarland.de/teaching/winter18-19/planning.html']

    def parse(self, response):
        url = response.url
        names = []
        pdf_urls = []

        for i in response.xpath('//a/@href'):
            pdf_url = i.extract()
            name = pdf_url.split('/')[-1]
            pdf_url = urllib.parse.urljoin(url, pdf_url)
            if name.endswith('.pdf'):
                names.append(name)
                pdf_urls.append(pdf_url)

        yield Slide(names=names,
                    file_urls=pdf_urls)
