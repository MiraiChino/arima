from contextlib import contextmanager

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
        def select(value):
            select = Select(self.wait_clickable_element(f"select#{selector_id}"))
            select.select_by_value(value)
        dropdown = self.find_element_by_css_selector(f"select#{selector_id}")
        values = [o.get_attribute("value") for o in Select(dropdown).options[1:]]
        for value in values:
            while True:
                try:
                    select(value)
                    break
                except:
                    select(value)
                    break
            yield value

@contextmanager
def driver():
    chrome = ChromeDriver()
    try:
        yield chrome
    finally:
        chrome.quit()


