# Data preprocessing for Pile

1. Tokenization (this step takes hours for running)
```
python tokenize_pile.py
```
This will also create '.bin' files for 'train', 'validation' and 'test' split for the overall dataset. 

2. Create '.bin' files for each domain in Pile
```
python prepare_pile_by_topic.py --topic "Pile-CC"
python prepare_pile_by_topic.py --topic "Github"
# ...
```
where the domains are 'Pile-CC', 'Github', 'StackExchange', 'HackerNews', 'Wikipedia (en)', 'ArXiv', 'DM Mathematics', 'PubMed Abstracts', 'PubMed Central', 'NIH ExPorter', 'EuroParl', 'PhilPapers', 'USPTO Backgrounds', 'FreeLaw', 'Gutenberg (PG-19)', 'Enron Emails'. 
