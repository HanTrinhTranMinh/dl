import fiftyone as fo
import fiftyone.zoo as foz

# 1. Cấu hình các thiết lập
output_dir = "./dataset"
splits = ["train", "valid", "test"] # RT-DETR cần train và val, test là tùy chọn
max_samples = 500 # Để test code, hãy để số nhỏ. Để huấn luyện thật, hãy xóa dòng này.

for split in splits:
    print(f"--- Đang tải và chuẩn bị cho: {split} ---")
    
    # 2. Tải dữ liệu từ Zoo
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split=split,
        label_types=["detections"], # RT-DETR chủ yếu dùng detections
        max_samples=max_samples,
    )

    # 3. Xuất dữ liệu ra định dạng COCO Detection (chuẩn JSON)
    # Định dạng này tạo ra file instances_split.json và thư mục data/ chứa ảnh
    dataset.export(
        export_dir=output_dir,
        dataset_type=fo.types.COCODetectionDataset,
        split=split,
    )

print(f"Hoàn tất! Dữ liệu đã sẵn sàng tại: {output_dir}")