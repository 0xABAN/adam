## Stack 

- Not too sure how fine tuning works, but probably PyTorch, uv, etc. 

## ETL

- Ingest post-2022 discord messages, filter out 1-word responses for higher quality. 
- Backtranslate using claude haiku 5.5 (more info on how messages looks like, how outputs should look, etc.) 

## Training 

- leverage Runpod GPUs ($100 in credits). More specifications on model size & cost are needed 

## Hosting 

- Modal on-demand GPUs 
- Apply reasonabel rate limits 
