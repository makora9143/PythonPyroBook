# 『StanとRでベイズ統計モデリング』のPython/Pyro実装

## 概要
このレポジトリは，松浦先生の「StanとRでベイズ統計モデリング」の[コード](https://github.com/MatsuuraKentaro/RStanBook)をpython/pyroで再現しています．

したがって，基本的には松浦先生のルールに則り，実装しています．

各章の番号.ipynb

* jupyter notebook（`X-Y.ipynb`）

### 各章のディレクトリの構成

* `input`:元々のレポジトリと同様にデータを格納
* `exercise`:練習問題に対するjupyter notebook

## Stanとpyroの記述の違い

Stanでは，
```stan
model {
  for (n in 1:N) {
    Y[n] ~ normal(mu, 1);
  }
  mu ~ normal(0, 100);
}
```
とするらしい．（松浦先生準拠）

pyroでは，
```python
def model(data):
    mu0 = torch.tensor(0.0)
    sigma0 = torch.tensor(100.0)

    mu = pyro.sample("latent_mean", dist.Normal(mu0, sigma0))

    with pyro.plate("data", data.size(0)):
        pyro.sample("obs", dist.Normal(mu, 1), obs=data)
```
となるっぽい？

## 進捗状況

- 4章
  - [x] MCMC
  - [ ] VI
  - [ ] リファクタリング
- 5章
  - [x] MCMC
  - [ ] VI
  - [ ] リファクタリング
- 7章
  - [x] MCMC
  - [ ] VI
  - [ ] リファクタリング
- 8章
  - [x] MCMC(ただし，確率変数が多すぎて実行不可能)
  - [ ] VI
  - [ ] リファクタリング
- 10章
  - [ ] MCMC
  - [ ] VI
  - [ ] リファクタリング
- 11章
  - [ ] MCMC
  - [ ] VI
  - [ ] リファクタリング
- 12章
  - [ ] MCMC
  - [ ] VI
  - [ ] リファクタリング



## ソースコードの実行環境
| ソフトやパッケージ名 | バージョン | 
|:-----------|:------------|
| python|3.7.1||
|pyro|dev/0.3.0+0c49858a| 
|pytorch|1.1.0a0+8683b75| 
|numpy | 1.15.4|
|scipy |1.2.0 |
|pandas|0.23.4 |
|matplotlib |3.0.2 |
|seaborn |0.9.0 |
|statsmodels |0.9.0|

