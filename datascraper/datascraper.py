
import scraperutils
#from bs4 import BeautifulSoup
import wikipedia

searched_links = set()

def get_contents(url):
    try:
        soup = scraperutils.get_soup(url)
        head = soup.h1
        wiki = wikipedia.page(head)
        # Extract the plain text content of the page
        text = wiki.content
        return text
    except:
        return ""

def get_wiki_links(url):
    soup = scraperutils.get_soup(url)
    if soup == None:
        return set()
    links = soup.find_all('a')

    searchable = set()
    for i in range(0, 20):
        link = links[i]
        href = link.get('href')
        if href is not None and '/wiki/' in href and '/wiki/Wikipedia:' not in href and '/wiki/File:' not in href and '/wiki/Help:' not in href and href not in searched_links:
            searched_links.add(href)
            searchable.add(f"https://en.wikipedia.org{href}")

    return searchable

def main(starter_url):
    recursive_scrape(starter_url)

def recursive_scrape(url):
    searchable = get_wiki_links(url)
    #with open('sample.txt', 'a', encoding="utf8") as file:
    #    file.write(get_contents(url))
    for link in searchable:
        print(link)
        recursive_scrape(link)
    return True

main("https://en.wikipedia.org/wiki/Mikl%C3%B3s_Ybl")



