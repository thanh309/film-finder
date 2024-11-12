from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver import Chrome, ChromeOptions
from bs4 import BeautifulSoup
import re
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_WORKERS = 5
ADBLOCK_PATH = 'resources/adblock.crx'


PART = 1



def scrape_user_ratings(uid: str, driver: Chrome) -> None | list[str]:
    url = f'https://www.imdb.com/user/ur{uid}/ratings/?view=compact&title_type=feature&sort=num_votes%2Cdesc'
    while True:
        driver.get(url)

        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, 'title'))
        )

        html_content = driver.page_source
        soup = BeautifulSoup(html_content, 'html.parser')

        title_tag = soup.find('title')
        if title_tag and "Error 503" in title_tag.text:
            print(f"503 error encountered for user {uid}. Retrying in 30 seconds...")
            sleep(30)  # Wait 30 seconds before retrying
            continue
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
            sleep(scroll_pause_time)

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

                    rating_tag = item.find(
                        'span', {'aria-label': re.compile(r'User rating: \d+')})
                    if rating_tag:
                        rating = re.search(r'User rating: (\d+)',
                                        rating_tag['aria-label']).group(1)
                        ratings_data.append(f'{uid},{fid},{rating}')

        return ratings_data


def create_driver() -> Chrome:
    options = ChromeOptions()
    # options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--blink-settings=imagesEnabled=false')
    options.add_argument('--load-extension ' + ADBLOCK_PATH)
    driver = Chrome(options=options)
    return driver


def main(user_ids):

    num_threads = min(len(user_ids), MAX_WORKERS)
    drivers = [create_driver() for _ in range(num_threads)]

    try:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            with open(f'resources/ratings/users_ratings_part{PART}.csv', 'w') as fw:
                future_to_user_id = {
                    executor.submit(scrape_user_ratings, uid, drivers[i % num_threads]): uid
                    for i, uid in enumerate(user_ids)
                }

                for future in as_completed(future_to_user_id):
                    user_id = future_to_user_id[future]
                    try:
                        results = future.result()
                        if results:
                            for result in results:
                                fw.write(result)
                                fw.write('\n')
                    except Exception as e:
                        print(f"Error scraping user {user_id}: {e}")

    finally:
        for driver in drivers:
            driver.quit()


if __name__ == '__main__':
    with open(f'resources/uids/user_ids_part{PART}.txt', 'r') as f:
        uids = [line.strip() for line in f]
    main(uids)
