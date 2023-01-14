from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium import webdriver
import requests
import time
import bs4
import os


def Scroll_To_Bottom():
    for x in range(1,6):
        driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
        time.sleep(4)


def downloader(n,chosen_URL,folder):

        response =requests.get(chosen_URL)
        if response.status_code==200:
            with open(os.path.join(folder,str(n-1)+".jpg"),'wb') as file:
                file.write(response.content)
    

options = Options()
options.add_argument("--headless") 
options.add_argument('--no-sandbox') 
options.add_argument('--disable-gpu')  
options.add_argument('start-maximized') 
options.add_argument('disable-infobars')
options.add_argument('--disable-software-rasterizer')
options.add_argument("--disable-extensions")
driver = webdriver.Chrome(chrome_options=options, executable_path=r'C:\Users\jere-\FinalYearProject\Dev\chromedriver.exe')


animal = input("What would you Like to Webscrape From Google Images: ")
if not os.path.isdir(animal):
        os.makedirs(animal)
        
img_URL = "https://www.google.com/search?q=%s&source=lnms&tbm=isch"%animal
driver.get(img_URL)


Scroll_To_Bottom()
Begin = input("Please Press any Key to Begin Web Scraping !...") # allows used to scroll page and load more containers
driver.execute_script("window.scrollTo(0,0);") #Returns to Top of Page to 


html_page_code = driver.page_source # This fetches the HTML of the Entire web Page
soup = bs4.BeautifulSoup(html_page_code,'html.parser')
container = soup.find_all('div',{'class':"isv-r PNCib MSM1fd BUooTd"}) # this identifies all the containers by their class ID


container_no = len((container))
print("No of Containers Found:",container_no)


for x in range(2,container_no+2):

    sel_path = "#islrg > div.islrc > div:nth-child(%s)"%(x)                                    # This gets the selector for each container
    image_prev_spath = "#islrg > div.islrc > div:nth-child(%s) > a.wXeWr.islib.nfEiy > div.bRMDJf.islir > img"%(x)  

  
    try:
     preview_image = driver.find_element(By.CSS_SELECTOR ,image_prev_spath)
     preview_URL = preview_image.get_attribute('src')
     driver.find_element(By.CSS_SELECTOR,sel_path).click()

    except NoSuchElementException:
        pass
    
    time_out_check = time.time()

    while True:
        try:

            full_res_sel = "#Sva75c > div > div > div.dFMRD > div.pxAole > div.tvh9oe.BIB1wf > c-wiz > div.nIWXKc.JgfpDb > div.OUZ5W > div.zjoqD > div.qdnLaf.isv-id.b0vFpe > div > a > img"
            Full_res_elem = driver.find_element(By.CSS_SELECTOR,full_res_sel)
            full_res_URL  = Full_res_elem.get_attribute('src')

        except NoSuchElementException:
            pass

        if full_res_URL !=preview_URL:
            print("Full Res Link Working \n")
            break
        else:
            #making a timeout if the full res image can't be loaded
            time_gone = time.time()

            if time_gone - time_out_check > 10:
                print("Timeout! Will download a lower resolution image and move onto the next one")
                break

    try:
        downloader(x,full_res_URL,animal)
        print("Downloaded element %s out of %s total. URL: %s" % (x, container_no + 2, full_res_URL))
    except:
        print("Couldn't download image no: %s" %(x))

