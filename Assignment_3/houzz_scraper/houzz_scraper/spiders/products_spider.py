import scrapy
import os



class ProductsSpider(scrapy.Spider):
    name = 'products'
    # all requested products first pages links to start with
    start_urls = [
        'https://www.houzz.com/products/coffee-tables',
        'https://www.houzz.com/products/side-tables-and-accent-tables',
        'https://www.houzz.com/products/console-tables',
        'https://www.houzz.com/products/plant-stands-and-tables',
        'https://www.houzz.com/products/coffee-table-sets',

        'https://www.houzz.com/products/beds',

        'https://www.houzz.com/products/armchairs-and-accent-chairs',
        'https://www.houzz.com/products/recliner-chairs',
        'https://www.houzz.com/products/chaise-lounge-chairs',
        'https://www.houzz.com/products/gliders',
        'https://www.houzz.com/products/rocking-chairs',
        'https://www.houzz.com/products/dining-chairs',
        'https://www.houzz.com/products/lift-chairs',
        'https://www.houzz.com/products/massage-chairs',
        'https://www.houzz.com/products/folding-chairs-and-stools',

        'https://www.houzz.com/products/sofas'
    ]

    # use it for having a readable output and control number of crawled data
    counter = 0

    #######################################    constants classes    ######################################
    # class of <div> which all products are in
    ITEMS_DIV_SELECTOR = 'div.hz-card.clearfix.hz-br__result-set'
    #  class of <a> which contains each product's page's url
    ITEMS_SELECTOR = 'a.hz-product-card__link'
    #  class of <a> which contains the next page url
    NEXT_PAGE_SELECTOR = 'a.hz-pagination-link.hz-pagination-link--next'
    # class name of the <span> which contains product's name's text
    PRODUCT_TITLE_SELECTOR = 'span.view-product-title'
    #  class name of <li> which each of the m contains a keyword of 'The product has been described as' section
    PRODUCT_DESCRIPTION_TAG_SELECTOR = 'li.product-keywords__word'
    # class name of the <img> which contains first image of the product in lage scale
    LARGE_IMAGE_SELECTOR = 'img.view-product-image-print.visible-print-block'

    ######################################################################################################
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.limit = int(kwargs.get('limit'))

    def parse(self, response, **kwargs):

        # first select the <div> where products items are in
        page = response.css(self.ITEMS_DIV_SELECTOR)[0]
        # select all products item url
        urls_list = page.css(self.ITEMS_SELECTOR + '::attr(href)').getall()

        # for each product's page url call self.product_parse(url) function (to get requested data from each page)
        for url in urls_list:
            if self.counter < self.limit:
                # join extracted url with current url
                url = response.urljoin(url)
                self.counter += 1

                yield scrapy.Request(url, callback=self.product_parse)

        next_page = response.css(self.NEXT_PAGE_SELECTOR + '::attr(href)').get()
        # if there is any next page
        if next_page is not None:
            if self.counter < self.limit:
                # print(next_page)
                # join extracted url with current url
                next_page = response.urljoin(next_page)

                # call this same function for next page
                yield scrapy.Request(next_page, callback=self.parse)

    def product_parse(self, response):

        # print which product is processing
        print('----------------------    {}    --------------------------'.format(self.counter))
        # get product name text
        product_name = response.css(self.PRODUCT_TITLE_SELECTOR + '::text').get()

        # get all the keywords as a list of strings
        tags_list = response.css(self.PRODUCT_DESCRIPTION_TAG_SELECTOR + '::text').getall()

        # LARGE_IMAGE_SELECTOR = 'img.view-product-image-print.visible-print-block'
        large_image_url = response.css(self.LARGE_IMAGE_SELECTOR + '::attr(src)').get()

        # a list which contains urls of 2 images of this product
        images_url = list()

        # add the first image which would be the large image in the page
        images_url.append(large_image_url)

        try:  # check if second image exist for this product
            # select 2nd image in small image in a column in the left side of the large image
            second_image_small_url = response.css('div.alt-images__thumb')[1].css(
                'img::attr(src)').get()

            # use this pattern to find large image for both first and second image

            # large structure: https://st.hzcdn.com/simgs/87d180980da9f6a4_4-2051/home-design.jpg
            # small structure: https://st.hzcdn.com/fimgs/87d180980da9f6a4_2051-w65-h65-b1-p0--.jpg

            # extracted pattern by comparing large and small image for existing large link,
            # apply the pattern to the 2nd image
            url1 = large_image_url[8:].split('/')

            url2 = second_image_small_url[8:].split('/')
            url = 'https://' + '/'.join(url1[0:2]) + '/' + '-'.join(
                url2[2].replace('_', '_4-').split('-')[:2]) + '/' + '/'.join(
                url1[3:])

            # add url for large form of the second image to images' url list
            images_url.append(url)
        except IndexError:  # this product has only one image
            pass 

        # to save it to json file
        yield {'product_name': product_name,
               'images_url_list': images_url,
               'description_tags': tags_list
               }


if __name__ == '__main__':
    # get limits of how many first data you want to crawl generally
    limit = int(input('please enter an int to be crawled data limitation\n'))
    # start crawling and save json file into data.json
    os.system('scrapy crawl products -a limit={} -o data.json'.format(limit))
