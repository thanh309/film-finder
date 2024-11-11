from selenium.webdriver import Chrome, ChromeOptions
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep

MAX_WORKERS = 10
WAIT_TIME = 90

def create_driver() -> Chrome:
    options = ChromeOptions()
    options.add_argument('--disable-dev-shm-usage')  
    driver = Chrome(options=options)
    return driver


def scrape_ids_from_url(url: str, driver: Chrome) -> set[str]:
    driver.get(url)
    try:
        expand_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(
                (By.XPATH, "//button[.//span[text()='All']]")
            )
        )
        driver.execute_script('arguments[0].click();', expand_button)
        sleep(WAIT_TIME)

    except:
        pass

    html_content = driver.page_source

    soup = BeautifulSoup(html_content, 'html.parser')
    author_links = soup.find_all('a', {'data-testid': 'author-link'})

    ids = set()
    for link in author_links:
        match = re.search(r'/user/ur(\d+)/', link['href'])
        if match:
            ids.add(match.group(1))
    return ids



with open("resources/title_ids.txt", "r") as f:
    urls = [f"https://www.imdb.com/title/tt{line.strip()}/reviews/?sort=review_volume%2Cdesc" for line in f]


drivers = [create_driver() for _ in range(MAX_WORKERS)]

try:
    all_ids = set()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = {executor.submit(
            scrape_ids_from_url, url, drivers[i % MAX_WORKERS]): url for i, url in enumerate(urls)
        }

        for future in as_completed(futures):
            ids = future.result()
            all_ids.update(ids)

    with open("resources/unique_ids.txt", "w") as f:
        for user_id in all_ids:
            f.write(f"{user_id}\n")

finally:
    for driver in drivers:
        driver.quit()
