import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

#split salary by blank space and select the numeric number
def splitSalary(salary):
    salaryList = []
    for value in salary:
        salaryStr = value.split(' ')
        if salaryStr[0] == 'Up':
            salaryList.append(salaryStr[2])
        elif salaryStr[0] == 'From':
            salaryList.append(salaryStr[1])
        else:
            salaryList.append(salaryStr[0])

    salarySer = pd.Series(salaryList)
    return salarySer


#delete ',','$' in salary
def cleanSalary(salary):
    salaryList = []
    for value in salary:
        temp = value.replace(',', '')
        salaryStr = temp.split('$')
        salaryList.append(float(salaryStr[1]))

    # print(salaryList)
    salarySer = pd.Series(salaryList)
    return salarySer


#convert salary in hour, day, month to year
def convert_to_year(salary):
    salaryList = []
    for value in salary:
        if (value < 200):  # hour
            salaryList.append(value * 8 * 22 * 12)
        elif (value < 600):  # day
            salaryList.append(value * 22 * 12)
        elif (value < 20001):  # month
            salaryList.append(value * 12)
        else:
            salaryList.append(value)
    salarySer = pd.Series(salaryList)
    return salarySer


#draw seaborn to check exceptional data
def draw_boxplot(dataFrame):
    f,ax=plt.subplots(figsize=(10,8))
    import seaborn as sns
    sns.boxplot(x='experience_filter', y='Salary', data=dataFrame, ax=ax)
    # sns.boxplot(y='Salary', data=dataFrame, ax=ax)
    plt.show()


#Fill null salary
def padding(final_cleaned_data):
    # final_cleaned_data = pd.read_csv("final_cleaned_data.csv")
    location_list = ['San Jose', 'New York', 'San Francisco', 'California']
    job_list = ['backend developer', 'front developer', 'full stack developer']
    experience_list = ['Entry', 'Mid', 'Senior']

    for location in location_list:
        for job_type in job_list:
            for experience in experience_list:
                media_tmp = final_cleaned_data[(final_cleaned_data['location_filter'] == location) \
                                               & (final_cleaned_data['job_title_filter'] == job_type) \
                                               & (final_cleaned_data['experience_filter'] == experience)][
                    'Salary'].median()
                # print(media_tmp)
                final_cleaned_data.loc[((final_cleaned_data['location_filter'] == location) \
                                        & (final_cleaned_data['job_title_filter'] == job_type) \
                                        & (final_cleaned_data['experience_filter'] == experience)
                                        & (final_cleaned_data['Salary'].isna())), "Salary"] = media_tmp
    final_cleaned_data.to_csv("data_fill.csv", index=False)


if __name__ == "__main__":
    data = pd.read_csv(sys.argv[1])
    data = data[["Company", "Title", "job_title_filter", "location_filter", "experience_filter", "Salary"]]
    #delete null salary
    data = data.dropna(subset=['job_title_filter', 'Salary'])
    data = data.reset_index(drop=True)
    data.loc[:, 'Salary'] = splitSalary(data.loc[:, 'Salary'])
    data.loc[:, 'Salary'] = cleanSalary(data.loc[:, 'Salary'])
    data.loc[:, 'Salary'] = convert_to_year(data.loc[:, 'Salary'])
    data.to_csv('dataset/cleaned_data.csv', index=False)
    print(data.describe())
    # check group, remove Outliers
    group = data.groupby(by=["location_filter", "job_title_filter", "experience_filter"])["Salary"].mean()
    print(group)

    draw_data = data[(data["location_filter"] == "San Francisco")
                     & (data["job_title_filter"] == "software developer")]
    draw_boxplot(draw_data)

    data = data.drop(index=data[(data["location_filter"] == "San Francisco") & (data["job_title_filter"] == "software developer")
                                & (data["experience_filter"] == "Mid")
                                & (data["Salary"] >= 350000)].index)
    data = data.reset_index(drop=True)

    draw_data = data[(data["location_filter"] == "San Jose")
                     & (data["job_title_filter"] == "data science")]
    draw_boxplot(draw_data)

    data = data.drop(index=data[(data["location_filter"] == "San Jose") & (data["job_title_filter"] == "data science")
                                & (data["experience_filter"] == "Mid")
                                & ((data["Salary"] >= 210000) | (data["Salary"] <= 70000))].index)
    data = data.reset_index(drop=True)

    data = data.drop(index=data[(data["location_filter"] == "San Jose") & (data["job_title_filter"] == "data science")
                                & (data["experience_filter"] == "Senior")
                                & (data["Salary"] >= 250000)].index)
    data = data.reset_index(drop=True)

    data.to_csv('dataset/train_data.csv', index=False)

    group = data.groupby(by=["location_filter", "job_title_filter", "experience_filter"])["Salary"].mean()
    print(group)

