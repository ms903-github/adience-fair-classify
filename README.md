# fairer classification on adience dataset
fair age classification on adience dataset.   

## usage   
run
```
mkdir dataset
mkdir result
python main.py ./config/config.yaml
```
`config.yaml`をいじることで各種パラメータが調整可能．`num_f(m)_sample`でtrainにおける男女のサンプル数を指定できる．   
結果は`result`以下に格納される．config内の`result_path`で結果を出力するディレクトリを指定．   

## todo
- 敵対的学習の追加   
