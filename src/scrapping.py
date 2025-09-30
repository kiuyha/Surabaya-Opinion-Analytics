import httpx
from lxml import html
import time
import random
from datetime import datetime, timezone
from urllib.parse import quote_plus
from .Config import log, Client
from typing import Tuple, Optional
from datetime import datetime

def safetly_extract_text(element: html, xpath: str, attribute: Optional[str] = None)-> Optional[str]:
    try:
        if attribute:
            return element.xpath(xpath)[0].attrib.get(attribute).strip()
        else:
            return element.xpath(xpath)[0].text_content().strip()
    except:
        return None

def exctract_id_tweet(tweet)-> str:
    tweet_link = safetly_extract_text(tweet, './/a[contains(@class, "tweet-link")]', attribute='href')
    tweet_id = tweet_link.split('/')[-1].split('#')[0]
    return tweet_id

def get_posted_at(tweet)-> datetime:
    time_string = safetly_extract_text(tweet, './/span[contains(@class,"tweet-date")]/a', attribute='title')
    return datetime.strptime(time_string.replace('·', ''), "%b %d, %Y %I:%M %p %Z")

def extract_new_tweets_and_next_link(html_content: str)-> Tuple[list, Optional[str]]:
    tree = html.fromstring(html_content)
    links = tree.xpath('.//div[contains(@class, "show-more")]/a')
    list_tweets = tree.xpath('.//div[contains(@class, "timeline-item")]')

    # we use list comprehension since it more fast than for loop
    scraped_tweets = [
        {
            'id': exctract_id_tweet(tweet),
            'fullname': safetly_extract_text(tweet, './/a[contains(@class, "fullname")]'),
            'username': safetly_extract_text(tweet, './/a[contains(@class, "username")]'),
            'text_content': safetly_extract_text(tweet, './/div[contains(@class, "tweet-content")]'),
            'posted_at': get_posted_at(tweet),
            'like_count': int((safetly_extract_text(tweet, './/span[contains(@class, "tweet-stat") and .//span[contains(@class, "icon-heart")]]') or '0').replace(',', '')),
            'comment_count': int((safetly_extract_text(tweet, './/span[contains(@class, "tweet-stat") and .//span[contains(@class, "icon-comment")]]') or '0').replace(',', '')),
            'retweet_count': int((safetly_extract_text(tweet, './/span[contains(@class, "tweet-stat") and .//span[contains(@class, "icon-retweet")]]') or '0').replace(',', '')),
            'quote_count': int((safetly_extract_text(tweet, './/span[contains(@class, "tweet-stat") and .//span[contains(@class, "icon-quote")]]') or '0').replace(',', '')),
        }
        for tweet in list_tweets
    ]

    # Check if there is duplicate by compare the id with id in database
    scraped_ids = [tweet['id'] for tweet in scraped_tweets if tweet['id']]
    try:
        response = Client.table('tweets').select('id').in_('id', scraped_ids).execute()
        existing_ids = {item['id'] for item in response.data}
    except Exception as e:
        log.error(f"An error occurred while checking Supabase: {e}")
        existing_ids = set()

    new_tweets = [tweet for tweet in scraped_tweets if tweet['id'] not in existing_ids]

    if links and len(links) > 0:
        # idk why nitter use "show-more" class for "show newest" button so we use the second link
        next_link = (links[1] if len(links) > 1 else links[0]).attrib.get('href')
    else:
        next_link = None
    return new_tweets, next_link
        
def scrap_nitter(search_query: str, depth: int = -1, time_budget: int = -1)-> list:
    """ This function is for scrapping tweets from twitter/X using Nitter
    Args:
        search_query (str): The query to search for
        depth (int): The depth of the search (default: -1 for infinite)
        time_budget (int): The time budget for the search in seconds (default: -1 for infinite)
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
    
    if time_budget != -1 and not isinstance(time_budget, int):
        raise TypeError("time_budget must be an integer")

    log.info(f"Scraping tweets for query: {search_query} with depth: {depth}")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Mobile Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.5',
        'Upgrade-Insecure-Requests': '1',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }
    next_link = f'?q={quote_plus(search_query)}&f=tweets'
    scraped_data = []
    start_time = time.time()

    with httpx.Client(headers=headers, http2=True, timeout=None) as client:
        index = 1
        while True:
            # Check the time budget first
            if time_budget != -1 and (time.time() - start_time) > time_budget:
                log.info(f"Time budget of {time_budget}s exceeded. Stopping gracefully.")
                break
            
            # Check if we've run out of pages
            if next_link is None:
                log.info("No more pages available. Stopping.")
                break
                
            # Check if we've hit the desired depth
            if depth != -1 and index > depth:
                log.info(f"Reached max depth of {depth}. Stopping.")
                break

            try:
                response = client.get(f'https://nitter.net/search{next_link}')
                html_content = response.text
                status_code = response.status_code
                if status_code == 200 and html_content:
                    list_tweets, next_link = extract_new_tweets_and_next_link(html_content)
                    scraped_data.extend(list_tweets)
                    log.info(f"reqs={index}, length of tweets: {len(list_tweets)}, status code: {response.status_code}, Next link: {next_link}")

                    # delay to make it look like a human, you could lower it but usually it give blank response in 15th requests
                    time.sleep(random.uniform(2, 5))
                    index += 1
                elif status_code == 200 and not html_content:
                    log.info(f"Get empty response, reqs={index}, status code: {response.status_code}, length: {len(html_content)}")
                    break
                elif status_code == 429:
                    log.warning(f"Hit rate limit (429). Headers: {response.headers}")
                    wait_seconds = None
                    
                    # Check if the server gave us instructions
                    if 'retry-after' in response.headers:
                        retry_value = response.headers['retry-after']
                        
                        # Case 1: The header is a number of seconds
                        if retry_value.isdigit():
                            wait_seconds = int(retry_value)
                            
                        # Case 2: The header is an HTTP-date
                        else:
                            try:
                                retry_date = datetime.strptime(retry_value, "%a, %d %b %Y %H:%M:%S %Z").replace(tzinfo=timezone.utc)
                                now = datetime.now(timezone.utc)
                                # Calculate the difference in seconds
                                wait_seconds = (retry_date - now).total_seconds()
                            except ValueError:
                                log.error(f"Could not parse 'Retry-After' date: {retry_value}")
                                break # Break if the date format is unexpected

                    # If the server gave no instructions, or if the wait time is negative
                    if wait_seconds is None or wait_seconds < 0:
                        log.warning("No valid 'Retry-After' header found. Breaking loop.")
                        break

                    remaining_time = time_budget - (time.time() - start_time)
                    if wait_seconds > remaining_time:
                        log.warning(
                            f"Wait time ({wait_seconds}s) exceeds remaining budget ({remaining_time:.0f}s). Stopping."
                        )
                        break

                    # Safely wait and continue the loop
                    log.info(f"Waiting for {wait_seconds:.2f} seconds as instructed by the server...")
                    time.sleep(wait_seconds)
                else:
                    log.info(f"Request failed with status code: {status_code}")
                    break
            except Exception as e:
                log.error(f"An error occurred: {e}")
                break
    
    log.info(f"Scraped {len(scraped_data)} tweets\n")
    return scraped_data