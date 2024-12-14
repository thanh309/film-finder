import time
import requests
import random
from bs4 import BeautifulSoup
from json import loads
from html import unescape
import pandas as pd
import os
from tqdm import tqdm

DATA_DIR = 'resources/data'
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) Gecko/20100101 Firefox/131.0',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:130.0) Gecko/20100101 Firefox/130.0',
    'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/109.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.5790.170 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15'
]

PART = 1
output_dir = f'{DATA_DIR}/split_film_data'


with open(f'{DATA_DIR}/split_fids/fids_part_{PART}.txt', 'r') as fr:
    fids = fr.read().split()

os.makedirs(output_dir, exist_ok=True)

def rfc4180_format(field):
    if isinstance(field, str):
        # Escape double quotes and wrap the field in quotes if necessary
        if any(c in field for c in [',', '"', '\n', '\r']):
            field = f'"{field.replace("\"", "\"\"")}"'
    return str(field)

def get_film_data(film_id: str):
    """Fetches film data from IMDb."""
    film_url = f'https://www.imdb.com/title/tt{film_id}'
    try:
        r = requests.get(
            film_url,
            headers={'User-Agent': random.choice(USER_AGENTS)}
        )
        if r.status_code != 200:
            print(f"Failed to fetch data for ID {
                  film_id}. Status code: {r.status_code}")
            return None

        soup = BeautifulSoup(r.text, 'html.parser').head
        ld_json_script = soup.find('script', type='application/ld+json')
        if not ld_json_script:
            print(f"No ld+json script found for ID {film_id}")
            return None

        json_data = loads(ld_json_script.string)
        return json_data

    except requests.exceptions.RequestException as e:
        print(f"Request failed for ID {film_id}: {e}")
        return None
    except Exception as e:
        print(f"An error occurred for ID {film_id}: {e}")
        return None


def process_film(fid: str):
    """Fetches film data and formats it as a CSV row."""
    json_film = get_film_data(fid)
    if not json_film:
        return None

    try:
        name = unescape(json_film.get('name', ''))

        description = unescape(json_film.get('description', ''))
        rating_count = json_film.get(
            'aggregateRating', {}).get('ratingCount', 0)
        rating_value = json_film.get(
            'aggregateRating', {}).get('ratingValue', 0.0)
        content_rating = json_film.get('contentRating', '')
        genre = unescape(','.join(json_film.get('genre', [])))
        keywords = unescape(json_film.get('keywords', ''))
        duration = int(pd.Timedelta(json_film.get(
            'duration', 'PT0S')).total_seconds())
        date_published = json_film.get('datePublished', '0000-01-01')
        actors = unescape(','.join([actor['name']
                          for actor in json_film.get('actor', [])]))
        directors = unescape(
            ','.join([director['name'] for director in json_film.get('director', [])]))
        image = json_film.get('image', '')

        fields = [
            fid,
            name,
            description,
            rating_count,
            rating_value,
            content_rating,
            genre,
            keywords,
            duration,
            date_published,
            actors,
            directors,
            image,
        ]

        return ','.join(rfc4180_format(field) for field in fields) + '\n'
    
    except Exception as e:
        print(f"Error processing data for ID {fid}: {e}")
        return None


def main(fids):
    """Writes film data to a CSV file sequentially with a delay between requests."""
    with open(f'{output_dir}/film_data_part{PART}.csv', 'w', encoding='utf-8') as fw:
        # Write the header
        fw.write('fid,name,description,ratingCount,ratingValue,contentRating,genre,keywords,duration,datePublished,actor,director,image\n')

        for fid in tqdm(fids):
            result = process_film(fid)
            if result:
                fw.write(result)
            time.sleep(4)

main(fids)