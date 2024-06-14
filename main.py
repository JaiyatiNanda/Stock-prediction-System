from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import  WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time

service = Service(executable_path='chromedriver.exe')
driver = webdriver.Chrome(service=service)

#get url
driver.get("https://finance.yahoo.com/quote/AAPL/history")

WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//table[contains(@class,"W(100%) M(0)")]//tbody/tr')))

historical_data =[]

rows = driver.find_elements(By.XPATH, '//table[contains(@class,"W(100%) M(0)")]//tbody/tr')
for row in rows:
    cols = row.find_elements(By.TAG_NAME, 'td')
    data = [col.text for col in cols]
    historical_data.append(data)

driver.quit()
#dataframe
df = pd.DataFrame(historical_data,columns=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

df.to_csv('historical_data.csv', index=False)
