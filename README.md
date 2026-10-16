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
`--no-pose` (nhanh hơn), `--no-sensor` (không đọc cảm biến nghiêng), `--port 8000`.

## Cách ước lượng khoảng cách

| Nguồn | Cách tính |
|---|---|
| Điểm chạm sàn | Tia từ chân box cắt mặt sàn (biết chiều cao + góc cúi camera) |
| Chiều cao vật | `Z = H / (tan góc_chân − tan góc_đỉnh)` với chiều cao trung bình theo lớp |
| Pose (người) | Thân (dọc) dùng công thức góc, vai/đầu/bề rộng box (ngang) dùng `f·L/pixel`; gộp trung bình trọng số nghịch đảo phương sai, bỏ outlier |
| Cảm biến nghiêng | Gia tốc kế màn hình laptop (Windows) → góc cúi camera tự động |

Ký hiệu trên ảnh: `<` vật trong vùng mù (ép vào vùng ĐỎ), `~` đã hiệu chỉnh/bị che, `*` ước lượng bằng pose.

## Hiệu chỉnh

Bấm **Hiệu chỉnh**, đặt một vật cách camera đúng 2 m, click vào điểm vật chạm sàn. Các kích thước cơ thể
(`TORSO`, `SHOULDER`, `HEAD`, `BOX_W`) ở đầu `radar.py` có thể chỉnh nếu đo thực tế lệch đều.

## Test

```bash
python tests/test_geom.py   # hình học camera, calib, phân vùng
python tests/test_pose.py   # ước lượng khoảng cách bằng pose
```

## Hạn chế

- Camera nhìn ngang hoặc ngửa lên thì vùng mù lớn (≈2.9 m ở 0°, cao 1.5 m); nên cúi ~20°.
- Chỉ nhận diện các lớp COCO (người, ghế, bàn, sofa...), chưa thấy tường/cửa/cột.
