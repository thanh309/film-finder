from bs4 import BeautifulSoup
import re

from concurrent.futures import ThreadPoolExecutor

from selenium.webdriver import Chrome, ChromeOptions

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from time import sleep
MAX_WORKERS = 10

def create_driver() -> Chrome:
    options = ChromeOptions()
    driver = Chrome(options=options)
    return driver


def scrape_ids_from_url(url: str) -> set[str]:
    driver = create_driver()
    driver.get(url)
    expand_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable(
            (By.XPATH, "//button[.//span[text()='All']]")
        )
    )
    driver.execute_script('arguments[0].click();', expand_button)
    sleep(90)

    html_content = driver.page_source
    driver.quit()

    soup = BeautifulSoup(html_content, 'html.parser')
    author_links = soup.find_all('a', {'data-testid': 'author-link'})

    ids = set()
    for link in author_links:
        match = re.search(r'/user/ur(\d+)/', link['href'])
        if match:
            ids.add(match.group(1))
    return ids


def scrape_multiple_urls(urls: list[str]) -> set[str]:
    all_ids = set()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = executor.map(scrape_ids_from_url, urls)
        for ids in results:
            all_ids.update(ids)
    return all_ids


with open("resources/title_ids.txt", "r") as f:
    urls = [f"https://www.imdb.com/title/tt{line.strip()}/reviews/?sort=review_volume%2Cdesc" for line in f]

users = scrape_multiple_urls(urls)


with open('resources/user_ids.txt', 'w') as f:
    for uid in users:
        f.write(f'{uid}\n')