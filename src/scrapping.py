import httpx
from lxml import html
import time
import random
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def scrap_nitter(search_query: str, depth: int = -1)-> list:
    """ This function is for scrapping tweets from twitter/X using Nitter
    Args:
        search_query (str): The query to search for
        depth (int): The depth of the search (default: -1 for infinite)
    Returns:
        list: A list of tweets
    Raises:
        ValueError: If search_query is None
        TypeError: If search_query is not a string
        TypeError: If depth is not an integer
    """

    if search_query is None:
        raise ValueError("search_query cannot be None")
    elif not isinstance(search_query, str):
        raise TypeError("search_query must be a string")

    if not isinstance(depth, int):
        raise TypeError("depth must be an integer")
    
    logging.info(f"Scraping tweets for query: {search_query} with depth: {depth}")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Mobile Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.5',
        'Upgrade-Insecure-Requests': '1',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }
    next_link = f'?q={search_query}&f=tweets'
    scraped_data = []

    def safetly_extract_text(element, xpath, get_text=True, attribute=None):
        try:
            if get_text:
                return element.xpath(xpath)[0].text_content().strip()
            else:
                return element.xpath(xpath)[0].attrib.get(attribute).strip()
        except:
            return None

    with httpx.Client(headers=headers, http2=True, timeout=None) as client:
        index = 1
        while((next_link is not None) and ((index <= depth) or (depth == -1))):
            try:
                response = client.get(f'https://nitter.net/search{next_link}')
                html_content = response.text
                status_code = response.status_code
                if status_code == 200 and html_content:
                    tree = html.fromstring(response.text)
                    links = tree.xpath('.//div[contains(@class, "show-more")]/a')
                    list_tweets = tree.xpath('.//div[contains(@class, "timeline-item")]')

                    # we use list comprehension since it more fast than for loop
                    scraped_data.extend([
                        {
                            'fullname': safetly_extract_text(tweet, './/a[contains(@class, "fullname")]'),
                            'username': safetly_extract_text(tweet, './/a[contains(@class, "username")]'),
                            'text_content': safetly_extract_text(tweet, './/div[contains(@class, "tweet-content")]'),
                            'posted_at': safetly_extract_text(tweet, './/span[contains(@class,"tweet-date")]/a', get_text=False, attribute='title'),
                            'like_count': int((safetly_extract_text(tweet, './/span[contains(@class, "tweet-stat") and .//span[contains(@class, "icon-heart")]]') or '0').replace(',', '')),
                            'comment_count': int((safetly_extract_text(tweet, './/span[contains(@class, "tweet-stat") and .//span[contains(@class, "icon-comment")]]') or '0').replace(',', '')),
                            'retweet_count': int((safetly_extract_text(tweet, './/span[contains(@class, "tweet-stat") and .//span[contains(@class, "icon-retweet")]]') or '0').replace(',', '')),
                            'quote_count': int((safetly_extract_text(tweet, './/span[contains(@class, "tweet-stat") and .//span[contains(@class, "icon-quote")]]') or '0').replace(',', '')),
                        }
                        for tweet in list_tweets
                    ])

                    if links and len(links) > 0:
                        # idk why nitter use "show-more" class for "show newest" button so we use the second link
                        next_link = (links[1] if len(links) > 1 else links[0]).attrib.get('href')
                    else:
                        next_link = None
                    logging.info(f"reqs={index}, length of tweets: {len(list_tweets)}, status code: {response.status_code}, Next link: {next_link}")

                    # delay to make it look like a human, you could lower it but usually it give blank response in 15th requests
                    time.sleep(random.uniform(2, 5))
                    index += 1
                elif status_code == 200 and not html_content:
                    logging.info(f"Get empty response, reqs={index}, status code: {response.status_code}, length: {len(html_content)}")
                    break
                elif status_code == 429:
                    logging.info(f"Request failed with status code: {status_code}")
                    # time.sleep(10)
                    break
                else:
                    logging.info(f"Request failed with status code: {status_code}")
                    break
            except Exception as e:
                logging.error(f"An error occurred: {e}")
                break
    
    logging.info(f"Scraped {len(scraped_data)} tweets\n")
    return scraped_data