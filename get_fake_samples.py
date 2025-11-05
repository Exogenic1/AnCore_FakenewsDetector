import pandas as pd

df = pd.read_csv('fakenews/full.csv')
fake_samples = df[df['label']==1].sample(5, random_state=42)

print('=== FAKE NEWS SAMPLES FROM YOUR DATASET ===\n')
for i, (idx, row) in enumerate(fake_samples.iterrows(), 1):
    print(f'\n--- Sample {i} ---')
    print(row['article'][:400])
    print('...')
