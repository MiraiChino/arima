from contextlib import contextmanager
from functools import wraps

from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait


class ChromeDriver(Chrome):

    def __init__(self, timeout=10):
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-setuid-sandbox")
        options.add_argument("--no-first-run")
        options.add_argument("--no-sandbox")
        options.add_argument("--no-zygote")
        options.add_argument("--single-process")
        self.wait = WebDriverWait(self, timeout)
        super().__init__(options=options)

        # ChromeDriverのバージョンを表示
        print(f"ChromeDriver version: {self.capabilities['chrome']['chromedriverVersion'].split(' ')[0]}")

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
        dropdowns = self.find_elements(By.CSS_SELECTOR, f"select#{selector_id}")
        if not dropdowns:
          raise ValueError(f"No dropdown found with id: {selector_id}")

        # 最初の<select>要素を選択
        dropdown = dropdowns[0]
        
        if dropdown.tag_name.lower() != "select":
            raise ValueError("Expected a <select> element.")

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
