from bs4 import BeautifulSoup

soup = BeautifulSoup(response.content, features="html.parser")
res = soup.find('span')