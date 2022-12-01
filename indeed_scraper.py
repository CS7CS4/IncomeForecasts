import csv
import logging
import warnings
import os
from bs4 import BeautifulSoup  
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time

# setting ignore as a parameter
warnings.filterwarnings('ignore')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.getLogger("selenium").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)


class Scraper():
    def __init__(self, keyword):
        self.keyword = keyword
        self.base_url = "https://ie.indeed.com/jobs?q=%s"%self.keyword
        self.ERROR_TEMPLATE = "A {0} exception occurred. Arguments:\n{1!r}"
    
    def scrape_html(self, url):
        try:
            options = Options()
            options.headless = True 
            #Hide GUI 								
            driver = webdriver.Firefox(service_log_path=os.path.devnull, options=options)
            driver.get(url)
            #Wait for page to load
            WebDriverWait(driver, 100).until(EC.url_contains(url))	
            raw_html = driver.page_source
            driver.close()
            return raw_html
        except Exception as E:
            logging.error(self.ERROR_TEMPLATE.format(type(E).__name__, E.args))
            return None
    
    # use beautiful soup to parse
    def parse_indeed_data(self, raw_html):
        try:
            job_list = []
            if raw_html:
                soup = BeautifulSoup(raw_html, features="html.parser")
                for job_card in soup.find_all(attrs={"class":"job_seen_beacon"}):
                    job_attributes = {}
                    #Job Title/Link
                    for title_card in job_card.find_all(name="h2", attrs={"class":"jobTitle"}):
                        for Title in title_card.find_all(name="span"):
                            job_attributes.update({"Title" : Title["title"]})
                        for Link in title_card.find_all(name="a", attrs={"class":"jcs-JobTitle"}):
                            job_attributes.update({"Link" : "indeed.com" + Link["href"]})

                    for company_info_card in job_card.find_all("div", attrs={"class":"company_location"}):
                        #company name
                        for company_name in company_info_card.find_all("a", attrs={"data-tn-element":"companyName"}):
                            job_attributes.update({"Company" : company_name.get_text()})
                        #location
                        for location in company_info_card.find_all("div", attrs={"class":"companyLocation"}):
                            job_attributes.update({"Location" : location.get_text()})

                    for meta_data_card in job_card.find_all("div", attrs={"class":"metadataContainer"}):
                        # salary
                        for salary_info_card in meta_data_card.find_all("div", attrs={"class":"salary-snippet-container"}):
                            for salary_info in salary_info_card.find_all("div", attrs={"class":"attribute_snippet"}):
                                job_attributes.update({"Salary" : salary_info.get_text()})
                        # type full/part time
                        for job_type_card in meta_data_card.find_all("div", attrs={"class":"metadata"}):
                            for job_type_info in job_type_card.find_all("div", attrs={"class":"attribute_snippet"}):
                                job_attributes.update({"JobType" : job_type_info.get_text()})

                    #Date Posted
                    for date_title_card in job_card.find_all("table", attrs={"class":"jobCardShelfContainer"}):
                        for Date in date_title_card.find_all("span", attrs={"class":"date"}):
                            job_attributes.update({"Date" : Date.get_text().replace("PostedPosted", "Posted")})


                    for job_description_card in job_card.find_all("table", attrs={"class":"jobCardShelfContainer"}):
                        for job in job_description_card.find_all("div", attrs={"class":"job-snippet"}):
                            uls = job.find_all("ul")
                            descriptions = [li.text for ul in uls for li in ul.findAll('li')]
                            description = "*".join(descriptions)
                            job_attributes.update({"Description" : description})
                    job_list.append(job_attributes)
            else:
                raise Exception("Missing HTML Data")
            return job_list
        except Exception as E:
            logging.error(self.ERROR_TEMPLATE.format(type(E).__name__, E.args))


    def save_to_csv(self, final_data, save_path):
        df_ori = pd.DataFrame()
        if os.path.exists(save_path):
            df_ori = pd.read_csv(save_path)
        print(len(df_ori))
        df = pd.DataFrame(final_data)
        df_save = pd.concat([df_ori, df])
        print(len(df_save))
        df_save.to_csv(save_path, index=False)
        logging.info(save_path + " generated")


    def execute(self, page_start, page_end):									
        raw_htmls = []
        
        for i in range(page_start, page_end): 						
            url = self.base_url + "&start=%d" % (i*10) + "&l=&vjk=2a4f6a0d2bb4bc8c"
            print(url)
            logging.info("Scraping Pass #" + str(i + 1) + " for " + "'" + str(url) + "' ...")
            raw_html = self.scrape_html(url)
            with open("indeed_html/%d.html" % i, 'w') as f:
                f.write(raw_html)
            raw_htmls.append(raw_html)
            time.sleep(10)

        print(len(raw_htmls))

        parse_htmls = []
        for i in range(0, len(raw_htmls)):			#Parses the data for the three instances
            logging.info("Parsing pass #" + str(i + 1) + " ...")
            parse_htmls.append(self.parse_indeed_data(raw_htmls[i]))
        
        self.save_to_csv(parse_htmls, "indeed_data/%s.csv"%self.keyword)



if __name__ == "__main__": 	#Reads in arguments
    job_keywords = "software+engineer"
    scraper = Scraper(job_keywords)
    scraper.execute(10, 68)
   
    # datas = []
    # for i in range(5, 10):
    #     with open("indeed_html/%d.html"%i, 'r') as f:
    #         raw_html = f.read()
    #         datas = datas + scraper.parse_indeed_data(raw_html)
    
    # scraper.save_to_csv(datas, "indeed_data/%s.csv"%job_keywords)