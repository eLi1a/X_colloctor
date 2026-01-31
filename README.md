# how to analyze PDF report
https://github.com/eLi1a/X_colloctor/blob/main/report_analysis_readMe.md

# how to analyze X posts after collection
https://github.com/eLi1a/X_colloctor/blob/main/X_analysis_readMe.md

----
# how to use X_collector

Scripts to collect and process X (Twitter) posts for a University of Eastern Finland (UEF) research project.

## What this repo does
- Collect posts from X for a given username using the X API v2
- Save results to a CSV with a consistent naming convention
- Designed for lightweight text analysis pipelines (preprocessing + downstream NLP)

---

## Requirements
- Conda (Anaconda / Miniconda)
- An X Developer account + API access (Bearer Token)
- Internet access

---

## Clone the repo
```bash
git clone https://github.com/eLi1a/X_colloctor.git
cd X_colloctor
```

## Create and activate the conda environment
```bash 
conda env create -f environment.yml
conda activate data_collection
```

## Configure your X API token
Create a file named .env in the project root:

`X_BEARER_TOKEN=YOUR_TOKEN_HERE`

Notes:

- .env is included in .gitignore, so it won’t be committed to GitHub.

- Do not wrap the token in extra quotes unless it contains special characters.

## Run the X collector script

This repo provides a CLI script that collects posts for one username and saves them to a CSV.

- Example: last 365 days
`python src/scripts/run.py --username storaenso --days 365`

- Example: last 30 days
`python src/scripts/run.py --username storaenso --days 30`

- Example: fixed time window (RFC3339)
`python src/scripts/run.py --username storaenso \
  --start 2025-01-01T00:00:00Z \
  --end   2025-12-31T23:59:59Z`

## Output files

By default, output is saved under data/ using:

X_posts_<username>_<YYYY-MM-DD>.csv

Example:

`data/X_posts_storaenso_2026-01-31.csv`


CSV columns include:

- x_handle, x_id, tweet_id, created_at, lang

- retweet_count, reply_count, like_count, quote_count

- text

## Common issues
### Rate limits / 429 Too Many Requests

If you hit a rate limit, the script automatically backs off and retries.
On the Basic plan, you may still run into monthly caps depending on how much you collect.

*“The id query parameter value is not valid”*

This happens if you pass a username into an endpoint that expects a numeric user ID.
The script resolves usernames to numeric IDs automatically, so normally you shouldn’t see this.

### Missing dotenv

If you see ModuleNotFoundError: No module named 'dotenv', install it:

`pip install python-dotenv`


# Ethics & data use

This project is intended for research use. Please follow:

- X Developer Policy / Terms

- Your institution’s guidelines and IRB/ethics requirements (if applicable)

- Store credentials securely and never commit tokens

## License

MIT License

Copyright (c) 2026 Li Sheng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
