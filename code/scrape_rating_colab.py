'''
# put the files into the Drive, and make a notebook in Colab with the following cells:

from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/film-finder

!pip install selenium

!pip install pyvirtualdisplay

!apt-get install -y xvfb

!python code/scrape_rating_colab.py
'''



from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver import Chrome, ChromeOptions
from bs4 import BeautifulSoup
import re
from time import sleep, time
from pyvirtualdisplay import Display

ADBLOCK_PATH = 'resources/adblock.crx'


PART = 5



def scrape_user_ratings(uid: str, driver: Chrome) -> None | str | list[str]:
    url = f'https://www.imdb.com/user/ur{uid}/ratings/?view=compact&title_type=feature&sort=num_votes%2Cdesc'
    driver.get(url)

    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.TAG_NAME, 'title'))
    )

    html_content = driver.page_source
    soup = BeautifulSoup(html_content, 'html.parser')

    title_tag = soup.find('title')
    if title_tag and "Error 503" in title_tag.text:
        return "503_error"
    if title_tag and 'Private list' in title_tag.text:
        return

    count_tag = soup.find(
        'li', {'role': 'presentation'}, string=re.compile(r'\d+ titles'))
    if not count_tag:
        return

    rating_count = int(re.search(r'(\d+) titles', count_tag.text).group(1))

    # if rating_count <= 20:
    #     return

    scroll_pause_time = 3
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        try:
            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);")
            sleep(scroll_pause_time)

            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
            
        except:
            break

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

    print(f'Done uid {uid}')

    return ratings_data


def create_driver():
    display = Display(visible=0, size=(1920, 1080))
    display.start()

    options = ChromeOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920x1080')
    driver = Chrome(options=options)

    return driver


def main(user_ids):
    driver = create_driver()

    try:
        with open(f'resources/ratings/users_ratings_part{PART}.csv', 'w') as fw:
            for uid in user_ids:
                while True:
                    data = scrape_user_ratings(uid, driver)
                    if data == '503_error':
                        print(f"Received 503 error for {uid}. Waiting 30 seconds before retrying... time={time()}")
                        sleep(30)
                    elif data:
                        for line in data:
                            fw.write(line)
                            fw.write('\n')
                        sleep(5)
                        break
                    else:
                        sleep(5)
                        break

    finally:
        driver.quit()


if __name__ == '__main__':
    with open(f'resources/uids/user_ids_part{PART}.txt', 'r') as f:
        uids = [line.strip() for line in f]
    main(uids)
