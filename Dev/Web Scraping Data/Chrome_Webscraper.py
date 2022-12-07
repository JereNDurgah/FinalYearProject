import bs4 
import os
import requests
import time
from bs4 import BeautifulSoup
from  selenium import webdriver


Diver_Path = r'C:\Users\jere-\FinalYearProject\Dev\chromedriver.exe'
driver =webdriver.Chrome(Diver_Path)

cat_URL = "https://www.google.com/search?q=cat&source=lnms&tbm=isch"

a = input("Waiting for Input")

driver.execute_script("window.scrollTo(0,0);")

page_html = driver.page_source
soup =bs4.BeautifulSoup(page_html,"html.parser")
container_find =soup.find_all('div',{'class':"isv-r PNCib MSM1fd BUooTD"})

cont_len =len(container_find)
print(cont_len)

#//*[@id="islrg"]/div[1]/div[1]/a[1]/div[1]/img
driver.find_element("src","""//*[@id="islrg"]/div[1]/div[1]/a[1]/div[1]/img""").click()