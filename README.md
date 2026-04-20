# 動物偵測（最簡版）

使用 YOLOv8 預訓練模型偵測圖片中的動物（COCO 類別：鳥、貓、狗、馬、羊、牛、象、熊、斑馬、長頸鹿）。

## 安裝

```powershell
cd d:\Python_Project\AutoDetectPhoto
pip install -r requirements.txt
```

## 使用

```powershell
python detect_animals.py 你的圖片.jpg
```

可選參數：

- `-o 輸出.png`：指定標註圖儲存路徑
- `-c 0.3`：信心度門檻（0～1，預設 0.25）

第一次執行會自動下載小型模型 `yolov8n.pt`。
