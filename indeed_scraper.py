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
from urllib import parse

# setting ignore as a parameter
warnings.filterwarnings('ignore')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.getLogger("selenium").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)


class Scraper():
    def __init__(self):
        # self.base_url = "https://indeed.com/jobs?q=%s"%self.keyword
        # self.base_url = "https://www.indeed.com/jobs?q=software+engineer+%24105%2C000&l=New+York%2C+NY&sc=0kf%3Aattr%28EVPJU%29attr%28EXSNN%7CFCGTU%7CHFDVW%7CQJZM9%7CUTPWG%252COR%29explvl%28MID_LEVEL%29jt%28fulltime%29%3B&vjk=aa017bf6af3d5bfb"
        # self.base_url = "https://www.indeed.com/jobs?q=software+engineer+%24105%2C000&l=New+York%2C+NY&sc=0kf%3Aattr%28EVPJU%29attr%28FCGTU%7CHFDVW%7CQJZM9%7CUTPWG%252COR%29explvl%28ENTRY_LEVEL%29jt%28fulltime%29%3B&vjk=a68cf81811f9d4bc"
        self.base_url = "https://www.indeed.com/jobs?"

        self.ERROR_TEMPLATE = "A {0} exception occurred. Arguments:\n{1!r}"


        self.html_dir = "html/"

        # self.salay_filter = None
        self.location_filter = None
        self.education_filter = None
        self.experience_filter = None
        self.job_title_filter = None

        # self.salay_filters = [("105000",parse.quote("$10,5000")), ("120000",parse.quote("$12,0000")),
        #                       ("130000",parse.quote("$13,0000")), ("150000",parse.quote("$15,0000")),
        #                       ("175000",parse.quote("$17,5000"))]

        # &l=San+Francisco
        self.location_filters = ["San+Jose", "New+York", "San+Francisco", "California", "Washington", "Seattle"]
        # &sc=0kf%3Aattr%28X62BT%29%3B&vjk=e2c652971aac7074
        # &vjk=7f804051c13e6f15
        self.education_filters = [("Bachelor","0kf:attr(FCGTU|HFDVW|QJZM9|UTPWG%2COR)"), ("Master","0kf:attr(EXSNN|FCGTU|HFDVW|QJZM9|UTPWG%2COR)")]
        self.experience_filters = [("Mid","explvl(MID_LEVEL);"), ("Senior","explvl(SENIOR_LEVEL);"), ("Entry","explvl(ENTRY_LEVEL);")]
        self.job_title_filters = ["backend+developer", "front+developer", "full+stack+developer"]


    # def get_salary_filter(self, location, ):
    #     if location == "San+Jose" and job:
    #         self.salay_filters = [115000, 125000, 140000, 165000, 185000]
    #     elif location == "New+York":
    #         self.salay_filters = [105000, 120000, 130000, 150000, 175000]
    #     elif location == "San+Francisco":
    #         self.salay_filters = [105000, 120000, 130000, 150000, 175000]
    #     elif location == "California":
    #         self.salay_filters = [105000, 120000, 130000, 150000, 175000]
    #     elif location == "Washington":
    #         self.salay_filters = [105000, 120000, 130000, 150000, 175000]
    #     elif location == "Seattle":
    #         self.salay_filters = [105000, 120000, 130000, 150000, 175000]

    def execute(self):
        # https://www.indeed.com/jobs?q=backend+developer+%2490%2C000&l=Seattle&sc=0kf%3Aattr%28FCGTU%7CHFDVW%7CQJZM9%7CUTPWG%252COR%29explvl%28ENTRY_LEVEL%29%3B&vjk=7f804051c13e6f15
        # https://www.indeed.com/jobs?q=backend developer $90,000&l=Seattle&sc=0kf:attr(FCGTU|HFDVW|QJZM9|UTPWG%2COR)explvl(ENTRY_LEVEL);&vjk=7f804051c13e6f15

        for job_title_filter in self.job_title_filters:
            self.job_title_filter = job_title_filter.replace("+", " ")
            for location in self.location_filters:
                self.location_filter = location.replace("+", " ")
                for experience in self.experience_filters:
                    self.experience_filter = experience[0]
                    experience_quote = experience[1]
                    for education in self.education_filters:
                        self.education_filter = education[0]
                        education_quote = education[1]
                        end = 30
                        if location in ["New+York", "San+Francisco", "California"]:
                            end = 60
                        for i in range(0, end):
                            url = self.base_url + "q=" + job_title_filter + "&l=" + location \
                            + "&sc=" + parse.quote(education_quote+experience_quote) \
                            + "&vjk=7f804051c13e6f15" + "&start=%d" % (i*10)
                            logging.info("Scraping Pass #" + str(i + 1) + " for " + "'" + str(url) + "' ...")
                            raw_html = self.scrape_html(url)
                            if raw_html is not None:
                                if not os.path.exists(self.html_dir):
                                    os.mkdir(self.html_dir)
                                file_name =" %s_%s_%s_%s.html" % (self.job_title_filter, self.location_filter, self.education_filter,
                                                                     self.experience_filter)
                                with open("%s/%s_%d.html" % (self.html_dir, file_name, i), 'w') as f:
                                    f.write(raw_html)


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

                    # add filter attributes
                    # job_attributes.update({"salary_filter": self.salay_filter})
                    job_attributes.update({"location_filter": self.location_filter})
                    job_attributes.update({"job_title_filter": self.job_title_filter})
                    job_attributes.update({"education_filter": self.education_filter})
                    job_attributes.update({"experience_filter": self.experience_filter})

                    job_list.append(job_attributes)
            else:
                raise Exception("Missing HTML Data")
            return job_list
        except Exception as E:
            logging.error(self.ERROR_TEMPLATE.format(type(E).__name__, E.args))


    def save_to_csv(self, final_data, save_path):
        order = ["Title", "job_title_filter", "Company", "Location", "location_filter", "Salary", "salary_filter", "experience_filter", "education_filter", "JobType", "Link", "Date", "Description"]
        df_ori = pd.DataFrame()
        if os.path.exists(save_path):
            df_ori = pd.read_csv(save_path)
        print(len(df_ori))
        df = pd.DataFrame.from_dict(final_data)
        # df.columns = ["Title", "job_title_filter", "Company", "Location", "location_filter", "Salary", "salary_filter", "experience_filter", "education_filter", "JobType", "Link", "Date", "Description"]
        # print(df.columns)
        df = df[order]
        df_save = pd.concat([df_ori, df])
        print(len(df_save))
        df_save = df_save.drop_duplicates(subset=["Link"], keep='last')
        df_save.to_csv(save_path, index=False)
        logging.info(save_path + " generated")


    # def execute(self, page_start, page_end, save_path):
    #     # raw_htmls = []
    #
    #     for i in range(page_start, page_end):
    #         # url = self.base_url + "&start=%d" % (i*10) + "&l=&vjk=2a4f6a0d2bb4bc8c"
    #         url = self.base_url + "&start=%d" % (i*10)
    #         logging.info("Scraping Pass #" + str(i + 1) + " for " + "'" + str(url) + "' ...")
    #         raw_html = self.scrape_html(url)
    #         if raw_html is not None:
    #             if not os.path.exists(self.html_dir):
    #                 os.mkdir(self.html_dir)
    #             with open("%s/%s_%d.html" % (self.html_dir, self.name_prefix, i), 'w') as f:
    #                 f.write(raw_html)
    #             # raw_htmls.append(raw_html)
    #         time.sleep(10)

        # print(len(raw_htmls))

        # parse_htmls = []
        # for i in range(0, len(raw_htmls)):			#Parses the data for the three instances
        #     logging.info("Parsing pass #" + str(i + 1) + " ...")
        #     parse_htmls.append(self.parse_indeed_data(raw_htmls[i]))
        
        # self.save_to_csv(parse_htmls, save_path)



if __name__ == "__main__": 	#Reads in arguments
    scraper = Scraper()
    scraper.execute()

    # datas = []
    # for i in range(start, end):
    #     with open("%s/%d.html"%(html_dir, i), 'r') as f:
    #         raw_html = f.read()
    #         datas = datas + scraper.parse_indeed_data(raw_html)
    #
    # scraper.save_to_csv(datas, save_path)

    # path = "indeed_data/ie_%s.csv"%job_keywords
    # df = pd.read_csv(path)
    # df = df.drop_duplicates(subset=["Link"], keep='last')
    # print(len(df))
    # df.to_csv("indeed_data/ie_%s_drop.csv"%job_keywords)