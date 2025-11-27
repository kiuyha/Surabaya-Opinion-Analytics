from typing import TypedDict, Optional

class SearchConfigDict(TypedDict):
    """A dictionary representing a single search configuration."""
    query: str
    depth: Optional[int]
    time_budget: Optional[int]

class ScrapedTweetDict(TypedDict):
    """A dictionary representing a single tweet."""
    id: str
    text_content: Optional[str]
    fullname: Optional[str]
    posted_at: Optional[str]
    username: Optional[str]
    like_count: Optional[int]
    retweet_count: Optional[int]
    comment_count: Optional[int]
    quote_count: Optional[int]