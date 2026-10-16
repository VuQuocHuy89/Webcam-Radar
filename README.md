# Webcam Radar

Biến webcam laptop thành radar 2D nhìn từ trên xuống: YOLO phát hiện người/đồ vật, ước lượng khoảng cách
bằng hình học camera, cảnh báo 3 vùng **XANH / VÀNG / ĐỎ** theo 3 cung **TRÁI / GIỮA / PHẢI**.

## Cài đặt

```bash
pip install -r requirements.txt
```

Model `yolo11n.pt` và `yolo11n-pose.pt` sẽ tự tải ở lần chạy đầu.

## Chạy

```bash
python web.py            # giao diện web: http://localhost:8000 (điện thoại cùng wifi: http://<IP-laptop>:8000)
python radar.py          # cửa sổ OpenCV, điều khiển bằng phím (xem đầu file)
```

Tùy chọn chính của `web.py`: `--cam 1`, `--video file.mp4`, `--hfov 70`, `--height 1.5`, `--pitch 25`,
`--no-pose`, `--no-depth` (nhanh hơn), `--depth-threads 4`, `--no-sensor` (không đọc cảm biến nghiêng), `--port 8000`.

## Cách ước lượng khoảng cách

| Nguồn | Cách tính |
|---|---|
| Điểm chạm sàn | Tia từ chân box cắt mặt sàn (biết chiều cao + góc cúi camera) |
| Chiều cao vật | `Z = H / (tan góc_chân − tan góc_đỉnh)` với chiều cao trung bình theo lớp |
| Pose (người) | Thân (dọc) dùng công thức góc, vai/đầu/bề rộng box (ngang) dùng `f·L/pixel`; gộp trung bình trọng số nghịch đảo phương sai, bỏ outlier |
| Depth (Depth Anything V2 Small) | Độ sâu tương đối → mét nhờ neo (vật đo tốt + sàn); đo thêm cho vật bị cắt chân, quét vật cản theo cột ảnh (thấy cả tường/cửa) |
| Kalman | Mỗi vật 1 track `[X, Z, vX, vZ]`, gộp mọi phép đo theo sai số từng cách, ghép bằng Mahalanobis |
| Cảm biến nghiêng | Gia tốc kế màn hình laptop (Windows) → góc cúi camera tự động |

Ký hiệu trên ảnh: `<` vật trong vùng mù (ép vào vùng ĐỎ), `~` đã hiệu chỉnh/bị che, `*` ước lượng bằng pose, `d` có dùng depth. Trên radar: chấm xám = điểm quét depth, mũi tên = vị trí dự đoán sau 1 s.

## Hiệu chỉnh

Bấm **Hiệu chỉnh**, đặt một vật cách camera đúng 2 m, click vào điểm vật chạm sàn. Các kích thước cơ thể
(`TORSO`, `SHOULDER`, `HEAD`, `BOX_W`) ở đầu `radar.py` có thể chỉnh nếu đo thực tế lệch đều.

## Test

```bash
python tests/test_geom.py   # hình học camera, calib, phân vùng
python tests/test_pose.py   # ước lượng khoảng cách bằng pose
python tests/test_kalman_depth.py   # Kalman + đổi depth ra mét
```

## Hạn chế

- Camera nhìn ngang hoặc ngửa lên thì vùng mù lớn (≈2.9 m ở 0°, cao 1.5 m); nên cúi ~20°.
- Depth cần ít nhất 1 vật đo tốt (người thấy rõ thân, hoặc sàn) làm neo mới đổi ra mét được.
- CPU: đủ 3 model (detect + pose + depth) ~4–5 FPS; tắt bớt bằng `--no-depth` / `--no-pose`.
