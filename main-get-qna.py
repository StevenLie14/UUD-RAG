import os
import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urljoin

BASE_URL = "https://www.hukumonline.com/klinik/arsip"
SHORT_ANSWER_FILE = "short_answer.json"
LONG_ANSWER_FILE = "long_answer.json"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
REQUEST_DELAY_SECONDS = 2

def load_json_file(filename):
    """Generic function to load a JSON file."""
    if not os.path.exists(filename):
        return []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Expecting a list of objects directly or {"questions": []}
            if isinstance(data, list):
                return data
            return data.get("questions", []) if isinstance(data, dict) else []
    except (json.JSONDecodeError, OSError) as exc:
        print(f"WARNING: Could not read {filename} ({exc}). Starting fresh.")
        return []

def persist_data(short_data, long_data):
    """Save both JSON files."""
    with open(SHORT_ANSWER_FILE, 'w', encoding='utf-8') as f:
        json.dump(short_data, f, indent=2, ensure_ascii=False)
    
    with open(LONG_ANSWER_FILE, 'w', encoding='utf-8') as f:
        json.dump(long_data, f, indent=2, ensure_ascii=False)

def clean_html_text(html_content):
    """Helper to remove HTML tags from text extracted via JSON blob."""
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text(separator="\n", strip=True)

def get_question_cards_from_page(page_url):
    print(f"\n--- Fetching list page: {page_url} ---")
    try:
        response = requests.get(page_url, headers=HEADERS, timeout=15)
        response.raise_for_status() 
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {page_url}: {e}")
        return [], False

    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Updated selector based on generic structure if classes change, 
    # but keeping your original class selector as it seems to work for the list page.
    question_cards = soup.find_all('div', class_="border-b-[0.5px] border-neutral-200 last:border-0 pb-4 md:pb-5 last:pb-0")
    
    detail_links = []
    for card in question_cards:
        link = card.find('a', href=True)
        if link:
            href = link.get('href')
            absolute_url = urljoin(BASE_URL, href)
            detail_links.append(absolute_url)
    
    print(f"Found {len(detail_links)} question cards on this page.")
    
    has_next_page = False
    # Robust check for next button
    next_buttons = soup.find_all('a', href=True)
    for btn in next_buttons:
        if '/page' in btn['href'] and 'arsip' in btn['href']:
            # Simple check: if the link page number > current page number logic (simplified here)
            # Or just assuming if a pagination link exists at bottom right
            has_next_page = True
            break
            
    return detail_links, has_next_page

def extract_qna_from_detail_page(detail_url):
    print(f"   -> Processing detail page: {detail_url}")
    try:
        response = requests.get(detail_url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"   -> Error fetching detail page {detail_url}: {e}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    
    # 1. Extract Question
    question_element = soup.find('h1', class_="mb-3 mt-1 css-wvkki6 e1qi4wgh0")
    if not question_element:
        # Fallback for question
        question_element = soup.find('h1')
        
    question_text = question_element.get_text(strip=True) if question_element else "Unknown Question"
    print(f"   -> Question: {question_text[:50]}...")

    short_answer_text = ""
    long_answer_text = ""

    # STRATEGY 1: Try to get data from Next.js hydration script (Most Reliable)
    # The HTML file provided shows the content is hidden in __NEXT_DATA__
    next_data_script = soup.find('script', id='__NEXT_DATA__', type='application/json')
    
    found_in_json = False
    if next_data_script:
        try:
            json_data = json.loads(next_data_script.string)
            # Navigate path: props -> pageProps -> dehydratedState -> queries -> [0] -> state -> data -> data
            # Note: The path might vary slightly, but usually resides in pageProps
            queries = json_data.get('props', {}).get('pageProps', {}).get('dehydratedState', {}).get('queries', [])
            
            for query in queries:
                data_node = query.get('state', {}).get('data', {}).get('data', {})
                if 'summary' in data_node and 'answer_raw' in data_node:
                    short_answer_text = clean_html_text(data_node.get('summary', ''))
                    long_answer_text = clean_html_text(data_node.get('answer_raw', ''))
                    
                    # If question was missing from H1, grab it here
                    if question_text == "Unknown Question":
                        question_text = data_node.get('question', 'Unknown Question')
                        if isinstance(question_text, str) and question_text.startswith('<p>'):
                             question_text = clean_html_text(question_text)

                    found_in_json = True
                    break
        except Exception as e:
            print(f"   -> Error parsing JSON data: {e}")

    # STRATEGY 2: Standard HTML parsing (Fallback)
    if not found_in_json:
        # 2. Extract Short Answer (Intisari Jawaban)
        # We look for the ID "INTISARI_JAWABAN", then find the next sibling div
        intisari_header = soup.find(id="INTISARI_JAWABAN")
        if intisari_header:
            # In your HTML, the text is inside a container nearby
            # Navigate up to the container, then find the text body
            # Based on image_6fa2f7.png and HTML structure
            container = intisari_header.find_parent('div').find_parent('div')
            if container:
                # The text is usually in the sibling of the header wrapper
                content_div = container.find('div', class_=lambda x: x and 'text-body' in x)
                if content_div:
                    short_answer_text = content_div.get_text(separator="\n", strip=True)

        # 3. Extract Long Answer (Ulasan Lengkap)
        # Based on image_6f9fd1.png, look for ID ULASAN_LENGKAP
        ulasan_header = soup.find(id="ULASAN_LENGKAP")
        if ulasan_header:
            # The content is usually in a 'wrapper-content' div following the header
            # We search for the next 'wrapper-content' class in the DOM
            wrapper_content = soup.find('div', class_='wrapper-content')
            
            # Ensure this wrapper comes AFTER the question, not the question itself
            # (Assuming the first wrapper-content is the question, second is answer)
            all_wrappers = soup.find_all('div', class_='wrapper-content')
            if len(all_wrappers) > 1:
                # Usually the last one or the one after ULASAN_LENGKAP
                long_answer_text = all_wrappers[-1].get_text(separator="\n", strip=True)
            elif len(all_wrappers) == 1:
                long_answer_text = all_wrappers[0].get_text(separator="\n", strip=True)

    # Final cleanup
    if not short_answer_text: 
        print("   -> WARNING: Short answer empty.")
    if not long_answer_text:
        print("   -> WARNING: Long answer empty.")

    # Create the two distinct objects
    short_obj = {
        "question": question_text,
        "answer": short_answer_text
    }

    long_obj = {
        "question": question_text,
        "answer": long_answer_text
    }
    
    return short_obj, long_obj

def run_scraper():
    # Load existing data
    short_data = load_json_file(SHORT_ANSWER_FILE)
    long_data = load_json_file(LONG_ANSWER_FILE)
    
    # Use a set of seen questions to avoid duplicates
    seen_questions = {item.get("question", "") for item in short_data}
    
    page_number = 1
    has_next = True
    new_items_count = 0

    try:
        while has_next:
            current_page_url = f"{BASE_URL}/page/{page_number}"
            detail_links, has_next = get_question_cards_from_page(current_page_url)

            if not detail_links:
                print(f"No question cards found on page {page_number}. Ending.")
                break

            for link in detail_links:
                result = extract_qna_from_detail_page(link)
                
                if result:
                    short_obj, long_obj = result
                    
                    if short_obj["question"] not in seen_questions:
                        short_data.append(short_obj)
                        long_data.append(long_obj)
                        seen_questions.add(short_obj["question"])
                        new_items_count += 1
                        
                        # Save progress immediately
                        persist_data(short_data, long_data)
                
                time.sleep(REQUEST_DELAY_SECONDS)

            if not has_next:
                print("No 'Next' page found. Ending.")
                break

            print(f"\n--- Finished page {page_number}. Cooling down... ---")
            time.sleep(REQUEST_DELAY_SECONDS * 2)
            page_number += 1

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Progress saved.")
        persist_data(short_data, long_data)

    persist_data(short_data, long_data)

    print(f"\n=======================================================")
    print(f"Scraping complete.")
    print(f"Total items collected: {len(short_data)}")
    print(f"New items this run: {new_items_count}")
    print(f"Saved to: {SHORT_ANSWER_FILE} and {LONG_ANSWER_FILE}")
    print(f"=======================================================")

if __name__ == '__main__':
    run_scraper()