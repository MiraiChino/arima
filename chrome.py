from contextlib import contextmanager

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


@contextmanager
def driver():
    service = Service(ChromeDriverManager().install())
    options = Options()
    options.add_argument('--headless')
    chrome = webdriver.Chrome(service=service, options=options)
    try:
        yield chrome
    finally:
        chrome.quit()

def wait_element(element_id, driver, timeout=10):
    return WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.ID, element_id))
    )

def wait_all_elements(driver, timeout=10):
    return WebDriverWait(driver, timeout).until(
        EC.presence_of_all_elements_located
    )
