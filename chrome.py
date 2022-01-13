from contextlib import contextmanager
from functools import wraps

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


class ChromeDriver(webdriver.Chrome):

    def __init__(self, timeout=10):
        service = Service(ChromeDriverManager().install())
        options = Options()
        options.add_argument('--headless')
        self.wait = WebDriverWait(self, timeout)
        super().__init__(service=service, options=options)

    def wait_all_elements(self):
        return self.wait.until(
            EC.presence_of_all_elements_located
        )

    def wait_clickable_element(self, selector):
        element = self.wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
        )
        return element

    def select_options(self, selector_id):
        @retry(10, verb=True)
        def select(value):
            select = Select(self.wait_clickable_element(f"select#{selector_id}"))
            select.select_by_value(value)
        dropdown = self.find_element_by_css_selector(f"select#{selector_id}")
        values = [o.get_attribute("value") for o in Select(dropdown).options[1:]]
        for value in values:
            select(value)
            yield value

def retry(num, verb=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(num):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    if verb:
                        print(f"retry{i}: {func.__name__}: {e.__class__.__name__}")
        return wrapper
    return decorator

@contextmanager
def driver():
    chrome = ChromeDriver()
    try:
        yield chrome
    finally:
        chrome.quit()


