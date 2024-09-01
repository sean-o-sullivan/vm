false_pairs_list = []
for index, context_row in tqdm(context_embeddings_df.iterrows(), total=context_embeddings_df.shape[0], desc='Generating false pairs', dynamic_ncols=True):
    context_embed = ast.literal_eval(context_row['anchor_embedding'])
    mask = df.iloc[:, :-2].apply(lambda row: np.allclose(row.tolist(), context_embed), axis=1)
    if not mask.any():
        print(f"Warning: No matching author found for context embedding at index {index}")
        continue
    
    author = df.loc[mask, 'author'].values[0]
    other_authors_data = df[df['author'] != author].drop(columns=['embedding_id', 'author'])
    if other_authors_data.empty:
        print(f"Warning: No other authors found for context embedding at index {index}")
        continue
    
    negative_embed = other_authors_data.sample(1).values.flatten().tolist()
    false_pairs_list.append({'anchor_embedding': context_row['anchor_embedding'],
                             'negative_embedding': str(negative_embed)})
false_pairs_df = pd.DataFrame(false_pairs_list)
