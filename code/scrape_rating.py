from selenium.webdriver import Chrome, ChromeOptions
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep

ADBLOCK_PATH = 'resources/adblock.crx'

def create_driver() -> Chrome:
    options = ChromeOptions()
    # options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--blink-settings=imagesEnabled=false')
    options.add_argument('--load-extension ' + ADBLOCK_PATH)
    driver = Chrome(options=options)
    return driver

driver = create_driver()
driver.get(
    'https://www.imdb.com/user/ur16619373/ratings/?view=compact&title_type=feature&sort=num_votes%2Cdesc'
)

sleep(30)
driver.quit()