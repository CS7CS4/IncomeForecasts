import pandas as pd
import numpy as py

def main():
    data = pd.read_csv('final_cleaned_data1.csv', encoding = 'utf-8')

    # local_types = ('San Jose', 'New York', 'San Francisco', 'California')
    # data = pd.DataFrame(local_types, columns = ['location_filter'])

    #one-hot
    local_oh = pd.get_dummies(data.location_filter, prefix = 'location')
    job_oh = pd.get_dummies(data.job_title_filter, prefix = 'jobTitle')

    data['Salary'] = data['Salary'].astype(float)
    data['Salary'] = data['Salary'].apply(lambda x: salary_(x))

    experience_map = {
            "Entry": 0,
            "Mid": 1,
            "Senior": 2
        }
    data['experience_filter'] = data['experience_filter'].map(experience_map)
    output = pd.concat([job_oh, local_oh, data[['experience_filter', 'Salary']]], axis = 1)

    # data.drop(columns = ['Title'], inplace = True)
    # data.drop(columns = ['Company'], inplace = True)
    # data.drop(columns = ['Location'], inplace = True)
    # data.drop(columns = ['JobType'], inplace = True)
    # data.drop(columns = ['Link'], inplace = True)
    # data.drop(columns = ['Date'], inplace = True)
    # data.drop(columns = ['Description'], inplace = True)
    # data.drop(columns = ['education_filter'], inplace = True)

    # order = ['job_title_filter', 'location_filter', 'experience_filter', 'Salary']
    # data = data[order]

    # data.loc[(data['location_filter'] == "San Jose"), "location_filter"] = 0


    output.to_csv('dataset.csv', index = None, encoding = 'utf-8')

def salary_(val):
    if val >= 114762.65297509829:
        val = 1
    else:
        val = 0
    return val

if __name__ == '__main__':
    main()