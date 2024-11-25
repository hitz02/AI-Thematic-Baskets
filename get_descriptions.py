import pandas as pd
from openai import OpenAI
from os import getenv
from tqdm import tqdm

# gets API Key from environment variable OPENAI_API_KEY
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=getenv('OPENROUTER_API_KEY'),
)

def get_company_description(ticker):

    prompt = f"Provide a business description for {ticker}, including its sector, theme, industry, and main products. in 200 words or less"

    completion = client.chat.completions.create(
        model="google/gemini-flash-1.5",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200
    )

    response = completion.choices[0].message.content
    return response.rstrip("\n")


if __name__ == "__main__":
    tqdm.pandas()

    # Fetch S&P 500 tickers from Wikipedia
    sp500_tickers = pd.read_html(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    )[0]['Symbol'].tolist()

    company_desc_df = pd.DataFrame()
    company_desc_df['Ticker'] = sp500_tickers

    company_desc_df['company_description'] = company_desc_df['Ticker'].progress_apply(
        get_company_description
    )