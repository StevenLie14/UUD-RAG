import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import urljoin

BASE_URL = "https://peraturan.go.id/cari?PeraturanSearch%5Btentang%5D=&PeraturanSearch%5Bnomor%5D=&PeraturanSearch%5Btahun%5D=&PeraturanSearch%5Bjenis_peraturan_id%5D=3&PeraturanSearch%5Bpemrakarsa_id%5D=&PeraturanSearch%5Bstatus%5D=Berlaku"
DOWNLOAD_DIR = "peraturan_pdfs"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
REQUEST_DELAY_SECONDS = 2

if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)
    print(f"Created download directory: {DOWNLOAD_DIR}")


def get_detail_page_links(page_url):
    print(f"\n--- Fetching list page: {page_url} ---")
    try:
        response = requests.get(page_url, headers=HEADERS, timeout=15)
        response.raise_for_status() 
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {page_url}: {e}")
        return [], False

    soup = BeautifulSoup(response.content, 'html.parser')
    link_elements = soup.select('a[href^="/files/"], a[href^="/uu-"]')
    unique_detail_links = set()
    for link in link_elements:
        href = link.get('href')
        absolute_url = urljoin(BASE_URL, href)
        unique_detail_links.add(absolute_url)

    detail_links = list(unique_detail_links)
    print("DETAIL LINKS:", detail_links)
    next_page_link = soup.select_one('a[href^="/cari?PeraturanSearch%5Btentang%5D=&PeraturanSearch%5Bnomor%5D=&PeraturanSearch%5Btahun%5D=&PeraturanSearch%5Bjenis_peraturan_id%5D=3&PeraturanSearch%5Bpemrakarsa_id%5D=&PeraturanSearch%5Bstatus%5D=Berlaku&page="]')
    has_next_page = bool(next_page_link)

    print(f"Found {len(detail_links)} detail links on this page.")
    return detail_links, has_next_page

def _download_file(pdf_url, filename):
    save_path = os.path.join(DOWNLOAD_DIR, filename)

    if os.path.exists(save_path):
        print(f"   -> File already exists: {filename}. Skipping download.")
        return

    print(f"   -> Downloading PDF: {filename}")
    try:
        pdf_response = requests.get(pdf_url, headers=HEADERS, stream=True, timeout=30)
        pdf_response.raise_for_status()

        if 'application/pdf' not in pdf_response.headers.get('Content-Type', ''):
            print(f"   -> WARNING: URL {pdf_url} did not return a PDF file. Skipping.")
            return

        with open(save_path, 'wb') as f:
            for chunk in pdf_response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"   -> SUCCESS: Downloaded and saved as {filename}")

    except requests.exceptions.RequestException as e:
        print(f"   -> Error downloading PDF from {pdf_url}: {e}")
        return
    
def download_pdf(link_url):
    print(f"   -> Processing link: {link_url}")

    if '/files/' in link_url and '.pdf' in link_url:
        print("   -> Direct PDF file link detected.")
        filename = link_url.split('/')[-1]
        
        valid_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '_', '-', '.')).rstrip()
        
        _download_file(link_url, valid_filename)
        return

    print("   -> Assuming HTML detail page.")
    try:
        response = requests.get(link_url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"   -> Error fetching detail page {link_url}: {e}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    pdf_image_element = soup.select_one('img[src="/img/icon/pdf2.png"]')
    pdf_link_element = None
    
    if pdf_image_element:
        pdf_link_element = pdf_image_element.find_parent('a')
    elif soup.select_one('a[href$=".pdf"]'):
         pdf_link_element = soup.select_one('a[href$=".pdf"]')


    if not pdf_link_element:
        print("   -> WARNING: Could not find a PDF download link (neither icon nor generic '.pdf' link). Skipping.")
        return

    pdf_href = pdf_link_element.get('href')
    pdf_url = urljoin(link_url, pdf_href)

    title_tag = soup.find('title')
    if title_tag:
        base_filename = title_tag.text.split('|')[0].strip()
        valid_filename = "".join(c for c in base_filename if c.isalnum() or c in (' ', '_', '-')).rstrip() + ".pdf"
    else:
        valid_filename = pdf_url.split('/')[-1]

    _download_file(pdf_url, valid_filename)

def run_scraper():
    page_number = 1
    has_next = True
    total_downloaded = 0
    while has_next or page_number <= 10000: 
        current_page_url = f"{BASE_URL}&page={page_number}"
        detail_links, has_next = get_detail_page_links(current_page_url)
        if not detail_links:
            print(f"No detail links found on page {page_number}. Ending scraping process.")
            break

        for link in detail_links:
            download_pdf(link)
            total_downloaded += 1
            time.sleep(REQUEST_DELAY_SECONDS)

        if not has_next:
            print("No 'Next' pagination link found. Ending scraping process.")
            break

        print(f"\n--- Finished page {page_number}. Waiting {REQUEST_DELAY_SECONDS * 5} seconds before next page... ---")
        time.sleep(REQUEST_DELAY_SECONDS * 5) 
        page_number += 1
    
    print(f"\n=======================================================")
    print(f"Scraping complete. Total files processed: {total_downloaded}")
    print(f"Check the '{DOWNLOAD_DIR}' folder for your PDFs.")
    print(f"=======================================================")


if __name__ == '__main__':
    run_scraper()
