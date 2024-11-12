from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver import Chrome, ChromeOptions
from bs4 import BeautifulSoup
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

UID = ('16619373', '0307374', '120934913')  # 290, 631, private
ADBLOCK_PATH = 'resources/adblock.crx'

def scrape_user_ratings(uid: str, driver: Chrome):
    url = f'https://www.imdb.com/user/ur{uid}/ratings/?view=compact&title_type=feature&sort=num_votes%2Cdesc'
    driver.get(url)

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, 'title'))
    )

    html_content = driver.page_source
    soup = BeautifulSoup(html_content, 'html.parser')

    title_tag = soup.find('title')
    if title_tag and 'Private list' in title_tag.text:
        return

    count_tag = soup.find(
        'li', {'role': 'presentation'}, string=re.compile(r'\d+ titles'))
    if not count_tag:
        return

    rating_count = int(re.search(r'(\d+) titles', count_tag.text).group(1))

    if rating_count <= 20:
        return

    scroll_pause_time = 2
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause_time)

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    html_content = driver.page_source
    soup = BeautifulSoup(html_content, 'html.parser')

    ratings_data = []
    items = soup.find_all(
        'li', class_='ipc-metadata-list-summary-item sc-4929eaf6-0 DLYcv cli-parent'
    )

    for item in items:
        link_tag = item.find('a', href=True)
        if link_tag:
            fid_match = re.search(r'/title/tt(\d+)', link_tag['href'])
            if fid_match:
                fid = fid_match.group(1)

                rating_tag = item.find('span', {'aria-label': re.compile(r'User rating: \d+')})
                if rating_tag:
                    rating = re.search(r'User rating: (\d+)', rating_tag['aria-label']).group(1)
                    ratings_data.append(f'{fid},{rating}')

    return ratings_data

def create_driver() -> Chrome:
    options = ChromeOptions()
    # options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--blink-settings=imagesEnabled=false')
    options.add_argument('--load-extension ' + ADBLOCK_PATH)
    driver = Chrome(options=options)
    return driver


driver = create_driver()

print(scrape_user_ratings(UID[0], driver))