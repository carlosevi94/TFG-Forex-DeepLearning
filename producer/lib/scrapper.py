from bs4 import BeautifulSoup
import requests
import random
import os


def get_random_user_agent():
    dir_path = os.path.join(os.path.dirname(__file__), os.path.pardir)
    with open(dir_path + "/resources/agents_lists.txt") as f:
        lines = f.readlines()
        res = random.choice(lines).strip()
    return res


def get_current_value(url):
    head = {
        "User-Agent": get_random_user_agent(),
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "text/html",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }
    response = requests.get(url, headers=head)
    soup = BeautifulSoup(response.content, features="html.parser")
    res = soup.find('span', attrs={'id': 'last_last'}).text
    return res
