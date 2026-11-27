# Tối ưu Định tuyến và Tiết kiệm Năng lượng trong Mạng IoT (PSO + Machine Learning)

**Phiên bản:** V19

## 1. Mục tiêu dự án
Mô phỏng, đánh giá và tối ưu định tuyến đa chặng tiết kiệm năng lượng cho mạng cảm biến không dây IoT (IEEE 802.15.4, 2.4 GHz) bằng thuật toán **Adaptive Particle Swarm Optimization (PSO)** kết hợp mô hình **Machine Learning (Gradient Boosting)** dự đoán chất lượng liên kết (PDR và độ trễ), so sánh đối chuẩn với **Shortest Path (SP)**, **Energy Dijkstra** và **PSO + Dijkstra**.

## 2. Kiến trúc dự án
```
├── main.py                     # Pipeline chạy quick mode & benchmark 5 seeds
├── evaluate_scalability.py     # Đánh giá mở rộng quy mô (N = 30, 50, 100)
├── requirements.txt            # Danh sách thư viện Python
├── src/
│   ├── config.py               # Cấu hình trung tâm (SimulationConfig)
│   ├── network.py              # Mô hình Node, IoTNetwork, đồ thị kết nối
│   ├── energy.py               # Mô hình năng lượng First-Order Radio (LEACH)
│   ├── channel.py              # Kênh truyền Log-distance + Log-normal shadowing + O-QPSK BER
│   ├── dataset.py              # Sinh và kiểm tra dataset huấn luyện ML (6.000 mẫu)
│   ├── ml_predictor.py         # Huấn luyện, đánh giá, xác thực và tải model ML
│   ├── routing.py              # GraphSnapshot, Reverse Dijkstra (heapq), RoutingEngine
│   ├── pso.py                  # Adaptive PSO tối ưu bộ trọng số w = (w_E, w_D, w_L, w_R)
│   ├── simulator.py            # Bộ mô phỏng theo vòng, phân mảnh IEEE 802.15.4, hàng đợi FIFO
│   └── visualizer.py           # Xuất bảng CSV, kiểm định thống kê ghép cặp, biểu đồ
├── tests/
│   └── test_all_modules.py     # Bộ kiểm thử tự động (80 tests)
├── data/                       # Dataset huấn luyện & OOD chuẩn
├── models/                     # Model ML đã huấn luyện (link_quality_predictor.joblib)
└── results/                    # Kết quả quick/, benchmark/, scalability/
```

## 3. Các thuật toán triển khai
1. **Shortest Path (SP):** Reverse Dijkstra tối thiểu hóa khoảng cách hình học `distance`.
2. **Dijkstra (Energy):** Reverse Dijkstra tối thiểu hóa tổng năng lượng liên kết `TX + RX + DA`.
3. **PSO + Dijkstra:** Adaptive PSO tối ưu bộ trọng số `w = (w_E, w_D, w_L, w_R)` trên các đặc trưng kênh vật lý để tối thiểu hóa hàm Fitness đa mục tiêu (năng lượng, phương sai năng lượng còn lại, độ trễ có điều kiện giao thành công, tỷ lệ mất gói; các node mất kết nối đóng góp PDR = 0 vào mẫu số).
4. **ML + PSO + Dijkstra:** Dùng mô hình **Gradient Boosting** dự đoán chất lượng liên kết vòng tiếp theo từ 7 đặc trưng nhân quả (`distance`, `prev_rssi_dbm`, `prev_snr_db`, `ewma_pdr`, `ewma_retx`, `ewma_delay_ms`, `rx_queue_load`) thu thập từ lịch sử truyền quá khứ, tích hợp cơ chế phát hiện OOD và fallback về mô hình phân tích vật lý khi liên kết chưa có lịch sử hoặc vượt ngưỡng phân phối.

## 4. Yêu cầu môi trường
- **Python:** `3.12.10` (khớp với `.venv` và môi trường chạy benchmark trong `experiment_config.json`)
- **Hệ điều hành:** Windows / Linux / macOS
- **Thư viện chính:** `numpy`, `pandas`, `scikit-learn==1.8.0`, `scipy`, `networkx`, `matplotlib`, `joblib`, `tqdm`, `psutil`, `pytest` (lưu ý: kết quả `ML + PSO` có thể sai lệch nhẹ nếu dùng phiên bản `scikit-learn` khác)

## 5. Cài đặt thư viện
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 6. Cách chạy mô phỏng
```bash
# Kiểm tra cú pháp toàn bộ mã nguồn
python -m compileall -q src main.py evaluate_scalability.py

# Chạy mô phỏng tùy chỉnh số vòng / seed / thư mục xuất
python main.py --rounds 100 --seeds 42 43 --output-dir results/custom
```

## 7. Cách chạy Quick Mode
Quick mode chạy smoke test nhanh (30 vòng, 1 seed `42`) và **cô lập hoàn toàn** dataset, model, dữ liệu OOD vào `results/quick/` (không ghi đè `data/`, `models/` hay `results/benchmark/`):
```bash
python main.py --quick
```

## 8. Cách chạy Benchmark chính thức
Chạy chuẩn 50 nút, 300 vòng, 5 seeds (`42, 43, 44, 45, 46`) cho 4 giao thức:
```bash
python main.py
```

## 9. Cách chạy Scalability
Đánh giá mở rộng quy mô ở `N = 30, 50, 100` nút (seed `42`):
```bash
# Quick mode (40 vòng)
python evaluate_scalability.py --quick

# Benchmark mode (150 vòng)
python evaluate_scalability.py
```

## 10. Cách chạy Test
```bash
python -m pytest tests -q
```

## 11. Vị trí lưu kết quả
- **Quick mode:** `results/quick/` (`comparison_raw.csv`, `comparison_summary.csv`, `ml_evaluation_metrics.json`, `experiment_config.json`, `charts/`, và artifact quick riêng)
- **Benchmark chính thức:** `results/benchmark/` (`comparison_raw.csv`, `comparison_summary.csv`, `comparison_paired_stats.json`, `ml_evaluation_metrics.json`, `runtime_profile.csv`, `experiment_config.json`, `charts/`)
- **Scalability:** `results/scalability_quick/` và `results/scalability/benchmark/` (`scalability_summary.csv`, `experiment_config.json`, `charts/scalability_analysis.png`)

## 12. Giới hạn thực tế của mô phỏng
- **Hiệu năng ML + PSO so với các baseline (kết quả chính thức):** Trong benchmark chính thức (50 nút, 300 vòng, 5 seeds) trên kênh tương quan thời gian AR(1) ($\rho = 0.75$):
  - **Tuổi thọ mạng (FND):** Nhóm định tuyến nhận biết chất lượng liên kết (**ML + PSO: 233.8 ± 39.7 vòng**, **PSO: 229.0 ± 38.3 vòng**) kéo dài FND gấp đôi so với **SP (115.2 ± 19.8)** và **Energy Dijkstra (99.0 ± 28.0)** (chênh lệch giữa `ML + PSO` và `PSO` là 4.8 vòng, $p = 0.63$, không có ý nghĩa thống kê).
  - **PDR & gói giao thành công:** **ML + PSO đạt 38.13 ± 4.89%** (`5.566 / 14.545` gói), cao hơn có ý nghĩa thống kê so với **PSO (35.11 ± 4.68%**, $p_{\text{holm}} = 0.0076$); so với `SP` (`36.13%`, $p_{\text{holm}} = 0.064$) và `Energy Dijkstra` (`35.54%`, $p_{\text{holm}} = 0.28$) chưa đạt ngưỡng $\alpha = 0.05$ sau hiệu chỉnh Holm.
  - **Năng lượng/gói & độ trễ:** `ML + PSO` đạt `1.352 ± 0.197 mJ/gói` (thấp hơn về mặt số học so với `1.455–1.619 mJ/gói`, nhưng $p_{\text{holm}} \ge 0.28$), độ trễ trung bình `38.85 ms` (so với `PSO: 37.60 ms`).
  - **Mô hình ML (Gradient Boosting):** Dự báo nhân quả đạt `PDR R² = 0.5704` (`RMSE = 0.3115`) và `Delay R² = 0.8126` (`RMSE = 9.8636 ms`).
- **Đặc điểm thuật toán PSO & ML trong mô phỏng:**
  - Hàm fitness của PSO mang tính hằng từng đoạn trên cây định tuyến rời rạc nên đường hội tụ phẳng ngay từ vòng lặp đầu trong benchmark, và sử dụng trạng thái kênh của vòng hiện tại.
  - Tỉ lệ fallback của `ml_pso` về mô hình phân tích vật lý đạt khoảng 79–82% (chỉ ~18–21% liên kết đã có lịch sử `link_history` dùng dự báo ML); thử nghiệm sơ bộ (3 seed) thay Gradient Boosting bằng persistence (`ewma_pdr`/`ewma_delay`) cho kết quả tương đương.
  - Cấu hình `noise_floor = -85 dBm` và chưa bổ sung baseline ETX/min-hop độc lập; thí nghiệm Scalability chạy trên 1 seed (`42`).
- **Dữ liệu ML tổng hợp:** Bộ dữ liệu huấn luyện (6.000 mẫu trên 9 topology) và OOD (1.000 mẫu trên 3 topology) được sinh từ mô phỏng chuỗi thời gian trên `WirelessChannel`, chưa sử dụng dữ liệu đo đạc phần cứng thực địa.
- **Năng lượng gói điều khiển:** Các hàm `control_tx_energy` / `control_rx_energy` đã được định nghĩa trong `src/energy.py` nhưng vòng lặp mô phỏng trong `src/simulator.py` chưa mô phỏng chi phí gói điều khiển/beacon.
- **Kiểm duyệt phải (Right-censoring) ở HND:** Với cấu hình 300 vòng và `initial_energy = 0.25 J`, `FND` đã ghi nhận đầy đủ cho cả 4 giao thức, nhưng mốc `HND` (50% nút chết) vẫn ở trạng thái `>300` (right-censored).
- **Mô phỏng rời rạc trên máy đơn:** Toàn bộ mạng chạy đồng bộ theo từng vòng trong bộ nhớ, chưa triển khai trên thiết bị IoT vật lý.
