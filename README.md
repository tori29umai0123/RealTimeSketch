# RealTimeSketch
RealTimeSketchは落書きをリアルタイムでイラスト化するソフトです。<br>
![1](https://github.com/tori29umai0123/RealTimeSketch/assets/72191117/057b46a2-e616-4dcf-bc45-af4279797b7e)

# Install
①以下のColabのリンクをクリックしてください<br>
 [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tori29umai0123/RealTimeSketch/blob/master/RealTimeSketch.ipynb)

②Colabのメニューから、『ランタイム』→『ランタイムのタイプを変更』→以下のように設定
![2](https://github.com/tori29umai0123/LineShadowGen/assets/72191117/f8cfa7ac-ed29-4353-bb0c-dd55a1a43137)

②Colabのメニューから、『すべてのセルを実行』（5分位かかります）
![3](https://github.com/tori29umai0123/LineShadowGen/assets/72191117/2eb56121-b061-4f26-9503-e078269fd27f)

④一番下の行のpublic URLをクリック

# Local Install
Python: [3.8.10](https://www.python.org/downloads/release/python-3810/) (It also worked on 3.10 series)  
CUDA Toolkit: [11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
①適当なディレクトリでリポジトリをgit clone<br>
```
cd C:/
git clone https://github.com/tori29umai0123/RealTimeSketch.git<br>
```
②install.ps1を右クリック→PowerShellで実行（5分位かかります）<br>
③RealTimeSketch_run.batをダブルクリックするとURLが表示されるのでそのURLをブラウザから開く<br>

# Local Upadate
ソフトの更新はRealTimeSketch_update.batをダブルクリックする。<br>
更新の有無を聞かれたら『y』と入力。（5分位かかります）<br>
モデルの更新はRealTimeSketch_update_model.batをダブルクリックする。<br>
モデルのアップデートを聞かれたら『y』と入力。（5分位かかります）。<br>

# Local Uninstall
フォルダごと削除してください

# Usage
prompt：画像の内容のタグ<br>
control weight：落書きの影響度
