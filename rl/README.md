# Ngã Tư Ảo · Giai đoạn 2: xe tự lái học bằng PPO

Một chiếc xe tự lái học cách qua một ngã tư đông xe máy. Đám xe máy, ô tô, xe buýt xung quanh chạy bằng lõi mô phỏng 2D (luật + tính cách riêng từng xe). Chỉ chiếc xe tự lái là học.

## Cấu trúc

```
web/sim.js     lõi mô phỏng giao thông (dùng chung cho trang web và huấn luyện)
web/env.js     môi trường RL: quan sát, hành động, phần thưởng; MLPPolicy để chạy mạng trên trình duyệt
web/render.js  đồ hoạ 2D
web/ui.js      giao diện trang web
web/build.py   gộp tất cả thành một file nga-tu-ao.html
rl/worker.js   tiến trình Node chạy nhiều bản môi trường, nói chuyện với Python qua stdin/stdout
rl/train.py    PPO bằng PyTorch
```

## Cài đặt (MacBook)

```bash
brew install node            # Node 18 trở lên
pip install torch numpy
```

## Huấn luyện

```bash
cd rl
python train.py --steps 1000000            # mặc định: số worker = số nhân CPU - 1, 4 môi trường mỗi worker
python train.py --steps 3000000 --workers 6 --envs 4   # máy mạnh hơn, học lâu hơn
```

Kết quả nằm trong `rl/runs/teacher/`:
- `step_0.pt`, `step_120000.pt`, `step_400000.pt`, `final.pt`: checkpoint PyTorch
- các file `.json` cùng tên: trọng số mạng để trang web chạy
- `log.json`: tỉ lệ qua được / va chạm / vượt đỏ theo số bước

Trên container 2 nhân CPU, 1 triệu bước mất khoảng 1 giờ. MacBook M-series có 8–10 nhân sẽ nhanh hơn nhiều (chọn `--workers 7`).

## Đánh giá

```bash
python train.py --eval runs/teacher/final.pt --episodes 200 --density 0.9
python train.py --eval runs/teacher/step_0.pt --random     # mốc so sánh: hành động ngẫu nhiên
```

## Đưa lên trang web

```bash
cd web && python build.py     # đọc rl/runs/teacher/*.json và gắn vào nga-tu-ao.html
```

## Bài toán

- **Quan sát (182 số):** tốc độ, vị trí trong làn, khoảng cách tới vạch dừng và lối ra, hướng đi (thẳng / trái / phải), độ cong đường phía trước, màu đèn và số giây đếm ngược, và 16 xe gần nhất trong bán kính 30 m (vị trí và vận tốc tương đối, hướng, kích thước, khe hở thật giữa hai thân xe). Đây là "trạng thái đặc quyền": học sinh nhìn camera ở Giai đoạn 4 sẽ không có.
- **Hành động (2 số liên tục, mỗi 0,2 giây):** ga/phanh trong [-1, 1], và vị trí ngang trong làn trong [-1, 1].
- **Phần thưởng:** +0,1 cho mỗi mét tiến, +5 khi qua hết ngã tư, −10 và dừng lượt khi va chạm, −6 khi vượt đèn đỏ, phạt nhẹ khi đi sát xe khác dưới 0,8 m, khi đổi ga/lái giật cục, khi vào cua quá nhanh, −2 khi đứng quá 90 giây.
- **Mạng:** chính sách MLP 128-128 tanh, hàm giá trị 256-256; chuẩn hoá quan sát và phần thưởng; GAE λ = 0,95, γ = 0,99, clip 0,2.

Mỗi lượt xe xuất phát cách vạch dừng 45 m trên một hướng ngẫu nhiên (50% đi thẳng, 25% rẽ phải, 25% rẽ trái). Mật độ giao thông thay đổi từ vắng tới đông.

## Bước tiếp theo

- Giai đoạn 3: cho đám xe máy tiến hoá (MAP-Elites) để tìm những kiểu tính cách làm xe tự lái lúng túng, có bộ lọc để vẫn giống người thật; huấn luyện lại xe tự lái trên những tình huống đó.
- Giai đoạn 4: học sinh nhìn camera trong CARLA học lại từ "thầy" này (Learning by Cheating).
