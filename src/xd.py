import pandas as pd

df = pd.read_parquet('corpus_LSM_esp/lsm_dataset.parquet')
df['speaker'] = df['intento'].astype(str).str.zfill(5).str[:2]

print('=== Videos y glosas por señante ===')
by_speaker = df.groupby('speaker').agg(
    n_videos=('glosa', 'count'),
    n_glosas=('glosa', 'nunique'),
).sort_index()
print(by_speaker.to_string())

print()
print('=== Glosas cubiertas por señante (cuántas de 249 grabó cada uno) ===')
pivot = df.pivot_table(
    index='speaker', columns='glosa',
    values='video_id', aggfunc='count', fill_value=0
)
print((pivot > 0).sum(axis=1).to_string())

print()
print('=== Cuántos señantes grabaron cada glosa ===')
coverage = df.groupby('glosa')['speaker'].nunique()
dist = coverage.value_counts().sort_index()
print(dist.to_string())
print(f'\nMedia señantes/glosa : {coverage.mean():.2f}')
print(f'Min señantes/glosa   : {coverage.min()}')
print(f'Max señantes/glosa   : {coverage.max()}')

print()
print('=== Glosas con menos de 5 señantes ===')
pocas = coverage[coverage < 5].sort_values()
print(f'{len(pocas)} glosas con < 5 señantes:')
print(pocas.to_string())