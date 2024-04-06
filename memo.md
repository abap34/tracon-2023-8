# 全体アイデア


- seen, unseenで分ける

## ユーザの特徴量(seen限定)

- target encoding も freequency encoding も効きそう. 
- textからうまく取りたい
  - 長さ、単語数 etcの統計量.
  - ネガポジとか？
- text の vec の平均とって user の vec を作り出す → userのvecとanimeのcosine類似度を計算




train['anime_point_mean'] = train['user'].map(train.groupby('user')['anime_point'].mean())
train['anime_point_std'] = train['user'].map(train.groupby('user')['anime_point'].std())

test_seen['anime_point_mean'] = test_seen['user'].map(train.groupby('user')['anime_point'].mean())
test_seen['anime_point_std'] = test_seen['user'].map(train.groupby('user')['anime_point'].std())


save_columns(train['anime_point_mean'], 'train', col_rename='anime_point_mean')
save_columns(test_seen['anime_point_mean'], 'test', col_rename='test_seen_anime_point_mean')

save_columns(train['anime_point_std'], 'train', col_rename='anime_point_std')
save_columns(test_seen['anime_point_std'], 'test', col_rename='test_seen_anime_point_std')