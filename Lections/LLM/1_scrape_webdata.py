# scrabe the website for links and save the website as PDFs
import os
import requests
from bs4 import BeautifulSoup
import pdfkit
from tqdm import tqdm


# -------------------- install wkhtmltopdf -------------------- #
# install wkhtmltopdf on your system https://wkhtmltopdf.org/downloads.html
# You may need to add the programm to path once with this code:
# def add_wkhtmltopdf_to_path():
#     # add wkhtmltopdf to your PATH
#     import os
#     import winreg as reg




#     # Path to the wkhtmltopdf bin directory
#     wkhtmltopdf_path = r"C:\Program Files\wkhtmltopdf\bin"

#     # Add to PATH for the current session
#     os.environ["PATH"] += os.pathsep + wkhtmltopdf_path

#     # Add to PATH permanently
#     try:
#         key = r"System\CurrentControlSet\Control\Session Manager\Environment"
#         with reg.OpenKey(reg.HKEY_LOCAL_MACHINE, key, 0, reg.KEY_ALL_ACCESS) as reg_key:
#             current_path, reg_type = reg.QueryValueEx(reg_key, "Path")
#             if wkhtmltopdf_path not in current_path:
#                 new_path = current_path + ";" + wkhtmltopdf_path
#                 reg.SetValueEx(reg_key, "Path", 0, reg_type, new_path)
#                 print("wkhtmltopdf added to PATH successfully!")
#             else:
#                 print("wkhtmltopdf is already in PATH.")
#     except PermissionError:
#         print("Permission denied. Run this script as administrator.")
#     except FileNotFoundError:
#         print("Registry key not found.")




# -------------------- Configuration -------------------- #

BASE_URL = "https://service.berlin.de/dienstleistungen/en/"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # directory of the current script file
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "berlin_servicespdfs")
NUM_PAGES_TO_SCRAPE = 99999  # Limit the number of pages to avoid overloading the server

# -------------------- Helper Functions -------------------- #

def get_all_service_links(base_url: str):
    """
    Scrape the base URL for all service-related links.

    Args:
        base_url (str): The URL to scrape.

    Returns:
        List[str]: List of full URLs to individual service pages.
    """
    try:
        response = requests.get(base_url)
        response.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Failed to fetch base URL: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    links = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/dienstleistung/" in href and "/en/" in href:
            full_url = requests.compat.urljoin(base_url, href)
            links.append(full_url)

    return list(set(links))  # Remove duplicates

def save_page_as_pdf(url: str, output_dir: str):
    """
    Save a given service page as a PDF.

    Args:
        url (str): URL of the service page.
        output_dir (str): Directory to save the PDF in.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        service_id = url.strip("/").split("/")[-2]
        output_path = os.path.join(output_dir, f"{service_id}.pdf")
        pdfkit.from_url(url, output_path)
    except Exception as e:
        print(f"[ERROR] Failed to save {url} as PDF: {e}")

# -------------------- Main Logic -------------------- #

def main():
    print("[INFO] Scraping service links...")
    links = get_all_service_links(BASE_URL)

    if not links:
        print("[ERROR] No service links found.")
        return

    # Limit the number of links for this run
    links = links[:NUM_PAGES_TO_SCRAPE]
    print(f"[INFO] Found {len(links)} service pages to download.")

    for link in tqdm(links, desc="Saving PDFs"):
        save_page_as_pdf(link, OUTPUT_DIR)

    print(f"[SUCCESS] Scraping completed. {len(links)} PDFs saved in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()
