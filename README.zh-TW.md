# End2End Autodrive Image-Steer

[English README](README.md)

這是一個以 Forza 為資料收集與測試環境的 image-to-steering imitation learning 專案：用遊戲快速收集駕駛影像與遙測資料，訓練端到端視覺控制模型，再透過虛擬 Xbox 控制器與方向盤回饋在遊戲中即時駕駛。

<p align="center">
  <a href="docs/assets/live-debug.mp4">
    <img src="docs/assets/live-debug-poster.jpg" alt="Forza live debug view" width="900">
  </a>
</p>

<p align="center">
  <a href="docs/assets/live-debug.mp4">觀看 9 秒 live debug 片段</a>
</p>

## 為什麼用遊戲

真實自駕資料昂貴、難標註，也很難在相同條件下重複實驗。賽車遊戲提供一個實用的學術沙盒：豐富視覺場景、不同光線、賽道幾何、輪胎極限、遙測資料，以及快速重置。

這個專案利用遊戲做 supervised end-to-end driving。目標不是宣稱從遊戲畫面就能直接做真實道路自駕，而是把資料收集做得更快、更安全、更可重複，同時保留足夠複雜的控制問題來研究 image-to-action policy。

Sony AI 的 GT Sophy 是重要動機：Nature 論文展示了 Gran Turismo 中的深度強化學習 agent 可以處理非線性賽車控制與多人競速策略，甚至達到冠軍級表現。本 repo 探索的是一個比較小但相關的問題：如果把遊戲當成資料引擎，一個實用的影像轉方向盤 pipeline 可以做到什麼程度？

## 研究脈絡

| 年份 | 工作 | 與本專案的關係 |
| --- | --- | --- |
| 1984 | CMU Navlab | Navlab 開啟了長期的相機式自動與輔助駕駛研究路線。 |
| 2004 | DAVE | 小型越野機器人使用相機輸入與端到端學習，從人類駕駛資料預測方向盤。 |
| 2016 | NVIDIA End to End Learning for Self-Driving Cars | CNN 從前視相機 raw pixels 直接輸出 steering；作者也觀察到模型在沒有道路輪廓標註下，自己學到有用的道路特徵。 |
| 2022 | Sony AI GT Sophy | Gran Turismo 中的深度強化學習證明賽車遊戲能支撐高性能、複雜規則下的控制研究。 |
| 2026 | This repo | 使用 Forza 建立快速資料收集與端到端 image steering 評估迴圈。 |

## Demo Clips

以下是從本機錄影剪出的 GitHub-safe 無聲短片。原始大影片保留在本機 `media/`，不進 Git。

| Live Debug | Long Run | Short Control |
| --- | --- | --- |
| <a href="docs/assets/live-debug.mp4"><img src="docs/assets/live-debug-poster.jpg" alt="Live debug clip poster" width="280"></a> | <a href="docs/assets/long-run.mp4"><img src="docs/assets/long-run-poster.jpg" alt="Long run clip poster" width="280"></a> | <a href="docs/assets/short-control.mp4"><img src="docs/assets/short-control-poster.jpg" alt="Short control clip poster" width="280"></a> |
| [MP4](docs/assets/live-debug.mp4) | [MP4](docs/assets/long-run.mp4) | [MP4](docs/assets/short-control.mp4) |

## 系統架構

```mermaid
flowchart LR
    A[Forza screen] --> B[Frame capture]
    B --> C[Resize and road crop]
    D[Forza UDP telemetry] --> E[Telemetry vector]
    C --> F[MobileNetV3-Small visual encoder]
    E --> G[Prediction head]
    F --> G
    G --> H[Steer, accel, brake]
    H --> I[PID and optional Kalman steering feedback]
    I --> J[Virtual Xbox controller]
    J --> K[Forza]
```

Runtime 會擷取遊戲畫面，套用與訓練時一致的 resize、crop 與 ImageNet normalization；如果 checkpoint 需要 telemetry，會將視覺特徵與遙測資料融合，最後透過 `vgamepad` 送出方向盤控制。Live debug window 會顯示 raw frame、model input、Grad-CAM++ overlay，以及 AI steering 和 UDP steering 的對照。

核心檔案：

- `forza_autodrive/drive.py`: live inference loop、debug window、hotkeys、controller output
- `forza_autodrive/model.py`: MobileNetV3-Small backbone 與 steering head
- `forza_autodrive/preprocess.py`: capture、resize、crop、normalization
- `forza_autodrive/telemetry.py`: Forza Dash UDP parser
- `forza_autodrive/steering_control.py`: PID correction 與 optional scalar Kalman filtering
- `load_data.ipynb`: training、validation plots、prediction inspection、Grad-CAM/Score-CAM analysis

## 可解釋性觀察

Notebook 中包含訓練後 steering model 的 Score-CAM/Grad-CAM 類型檢查。

最後層 attention 比較偏高階、區域也比較粗：

<p align="center">
  <img src="docs/assets/scorecam-final-layer.jpg" alt="Final-layer Score-CAM montage" width="900">
</p>

淺層 attention 更局部，反覆打在路面紋理、車道線、護欄與賽道邊界：

<p align="center">
  <img src="docs/assets/scorecam-shallow-layer.jpg" alt="Shallow-layer Score-CAM montage highlighting road markings" width="900">
</p>

這和 NVIDIA 端到端自駕論文中的觀察相呼應：只用 steering supervision，CNN 也可能學到道路相關內部特徵，而不是必須先明確標註 lane 或 road outline。在這個 repo 中，這只是對 Forza 模型注意區域的定性 sanity check，不是因果性的正式證明。

## Quick Start

### 1. Runtime setup

先安裝 runtime dependencies。PyTorch 指令可依你的 CUDA 版本調整。

```powershell
py -m pip install torch torchvision numpy pillow opencv-python dxcam keyboard vgamepad pytorch-grad-cam
```

Windows live control 需要安裝或修復 ViGEmBus driver，`vgamepad` 才能建立虛擬 Xbox 360 controller。

### 2. 放置模型權重

Checkpoint 檔案大於一般 GitHub repo 適合追蹤的大小，所以預設忽略。請把本機權重放在：

```text
best_model.pth
```

### 3. 啟用 Forza telemetry

在 Forza 開啟 Data Out，使用 Dash format，並把 UDP 封包送到這台電腦：

```text
Host: 127.0.0.1 或你的 PC IP
Port: 9999
Format: Dash
```

### 4. 先 dry-run

這會執行 inference 與 debug window，但不控制虛擬手把。

```powershell
py -m forza_autodrive.drive --model best_model.pth --no-controller --debug-window
```

### 5. Live driving

確認遊戲視窗、telemetry 與 emergency stop 都正常後再使用。

```powershell
py -m forza_autodrive.drive --model best_model.pth --start-armed
```

Hotkeys:

- `F9`: arm/disarm AI control
- `F8`: emergency stop
- `Ctrl+C`: exit and reset controller

常用選項：

```powershell
py -m forza_autodrive.drive --steer-scale 2.0 --steer-feedback pid --steer-kalman --fps 30
```

## README Asset Workflow

原始錄影 `media/` 和模型 checkpoint 都刻意忽略。要發布到 README，只追蹤壓縮後的小素材：

```powershell
py -m pip install -r requirements-readme-assets.txt
py tools/export_readme_assets.py
```

Exporter 會：

- 從 `media/` 剪短無聲 clips
- 移除每段選取片段的尾端，並避開每個 source recording 的最後 1 秒
- 讓 MP4 目標小於 10 MiB
- 輸出小於 500 KiB 的 poster JPG
- 從 `load_data.ipynb` 擷取 compact Score-CAM montages

GitHub 一般 Git repo 中超過 50 MiB 會警告，超過 100 MiB 會被阻擋，所以 `media/` 與 `*.pth` 保留本機，除非改用 Releases 或 Git LFS 發布。

## References

- [CMU Navlab](https://www.cs.cmu.edu/afs/cs/project/alv/www/)
- [DAVE: Autonomous Off-Road Vehicle Control using End-to-End Learning](https://cs.nyu.edu/~yann/research/dave/)
- [NVIDIA: End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316)
- [NVIDIA paper PDF](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
- [Sony AI GT Sophy announcement](https://ai.sony/news/sonyai009)
- [Nature: Outracing champion Gran Turismo drivers with deep reinforcement learning](https://www.nature.com/articles/s41586-021-04357-7)
- [GitHub Docs: About large files on GitHub](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
