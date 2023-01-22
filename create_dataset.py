import os
import pandas as pd
import numpy as np 


# Init - * - * - * - * - * - * - * - * - * - * - * - *
from dotenv import load_dotenv
load_dotenv()
ENV = os.getenv('ENV')

# Helpers - * - * - * - * - * - * - * - * - * - * - * 
def add_row_num(df: pd.DataFrame, cname="row_number") -> pd.DataFrame: 
    df[cname] = np.arange(len(df))
    return df

# Create Datasets - * - * - * - * - * - * - * - * - * 

## Google Analytics Datasets
ga_daily_visits_df = pd.read_csv(
    'data/data-export.csv', 
    nrows=365,
    header=8
) 

ga_avg_eng_time_df = pd.read_csv(
    'data/data-export.csv', 
    nrows=365,
    header=744
) 

ga_all_df = (
    pd.merge(
        left=ga_daily_visits_df, 
        right=ga_avg_eng_time_df, 
        on="Nth day"
    )
    .pipe(pd.DataFrame.rename, columns={
        "Users": "GA number of users", 
        "Average engagement time": "GA Avg engagement time"
    })
)


## Linkedin Dataset
li_followers_df = (
    pd.read_excel('data/dotdash_followers.xls')
    .pipe(add_row_num, cname="Nth day")
)

li_visitors_df = (
    pd.read_excel('data/dotdash_visitors.xls')
    .pipe(add_row_num, cname="Nth day")
)

li_post_df = (
    pd.read_excel(
        'data/dotdash_content.xls', 
        header=1, 
        sheet_name=1
    )
)

li_content_df = (
    pd.read_excel('data/dotdash_content.xls', header=1)
    .pipe(add_row_num, cname="Nth day")
    .pipe(pd.merge, right=li_post_df, how='left', left_on='Date', right_on='Created date')
)

li_all_df = (
    pd.merge(
        left=li_content_df,
        right=li_visitors_df, 
        on=["Nth day", "Date"],
        how="left"
    )
    .pipe(
        pd.merge,
        right=li_followers_df, 
        on=["Nth day", "Date"],
        how='left'
    )
)



## Output dataset
output_df = (
    li_all_df.merge(
        right=ga_all_df, 
        how='left',
        on="Nth day"
    )
)

output_df.to_parquet('data/li_ga_prediction.parquet')
