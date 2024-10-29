import requests
import random
from bs4 import BeautifulSoup
from json import loads
from html import unescape

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) Gecko/20100101 Firefox/131.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:130.0) Gecko/20100101 Firefox/130.0',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/109.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.5790.170 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15'
]


def get_film_data(id: str):
    film_url = f'https://www.imdb.com/title/tt{id}'
    r = requests.get(
        film_url,
        headers={'User-Agent': random.choice(USER_AGENTS)}
    )
    soup = BeautifulSoup(r.text, 'html.parser').head
    ld_json_script = soup.find('script', type='application/ld+json')
    json_data = loads(ld_json_script.string)
    print(json_data["aggregateRating"]["ratingCount"])

if __name__ == '__main__':
    # test film
    film_id = '0910970'
    print(get_film_data(film_id))
