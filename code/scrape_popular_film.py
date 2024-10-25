from bs4 import BeautifulSoup
import re

from selenium.webdriver import Chrome
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from time import sleep

MAX_ITER = 100

driver = Chrome()
driver.get(
    'https://www.imdb.com/search/title/?title_type=feature,tv_series&sort=num_votes,desc'
)
expand_button = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable(
        (By.XPATH, "//button[.//span[text()='50 more']]")
    )
)
for i in range(MAX_ITER - 1):
    driver.execute_script("arguments[0].click();", expand_button)
    sleep(5)


page_source = driver.page_source
soup = BeautifulSoup(page_source, 'html.parser')

all_links = list(set(
    re.search(r'/title/tt(\d+)', a['href']).group(1)
    for a in soup.find_all('a', href=True) if a['href'].startswith('/title')
))

with open('resources/title_id.txt', 'w') as f:
    for link in all_links:
        f.write(f"{link}\n")


driver.quit()