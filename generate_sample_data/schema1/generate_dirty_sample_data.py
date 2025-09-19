import pandas as pd
import uuid
import random
from datetime import datetime, timedelta
import faker
import os
import numpy as np

fake = faker.Faker()

output_dir = "telecom_cdr_csv"
os.makedirs(output_dir, exist_ok=True)

# ---------------- Hàm làm bẩn ----------------
def make_dirty(df_pd, dirty_frac=0.3, seed=42):
    rng = np.random.default_rng(seed)
    df = df_pd.copy()
    n_dirty = int(len(df) * dirty_frac)
    if n_dirty == 0:
        return df

    # Loại bỏ record_id khỏi danh sách cột được làm bẩn
    available_features = [c for c in df.columns if c != "record_id"]
    k = min(6, len(available_features))  # chọn ngẫu nhiên một số cột để làm bẩn
    features_to_dirty = rng.choice(available_features, size=k, replace=False)

    for col in features_to_dirty:
        idx = rng.choice(df.index, size=n_dirty, replace=False)

        for row_idx in idx:
            if rng.random() < 0.5:
                # thay bằng null
                df.at[row_idx, col] = None
            else:
                if pd.api.types.is_object_dtype(df[col]):
                    val = str(df.at[row_idx, col]) if df.at[row_idx, col] is not None else ""
                    dirty_suffix = "_" + "".join(rng.choice(list("ABCXYZ0123456789"), size=4))
                    df.at[row_idx, col] = val + dirty_suffix
                elif pd.api.types.is_numeric_dtype(df[col]):
                    val = df.at[row_idx, col]
                    if val is not None:
                        noise = rng.normal(loc=50, scale=20)  # thêm nhiễu
                        df.at[row_idx, col] = round(val + noise, 2)

    return df



# ---------------- Hàm sinh dữ liệu ----------------
def generate_telecom_data(start_time, end_time, n_records, apply_dirty=False):
    data = []
    for _ in range(n_records):
        start_ts = start_time + timedelta(
            seconds=random.randint(0, int((end_time - start_time).total_seconds()))
        )

        service_type = random.choice(["VOICE", "SMS", "DATA"])
        duration, sms_count, data_volume_mb = 0, 0, 0.0

        if service_type == "VOICE":
            duration = random.randint(1, 3600)
        elif service_type == "SMS":
            sms_count = random.randint(1, 5)
        elif service_type == "DATA":
            data_volume_mb = round(random.uniform(0.1, 500.0), 2)

        end_ts = start_ts + timedelta(seconds=duration if duration > 0 else random.randint(1, 60))

        row = {
            "record_id": str(uuid.uuid4()),
            "msisdn": f"+84{random.randint(700000000, 999999999)}",
            "service_type": service_type,
            "call_start": start_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "call_end": end_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": duration,
            "sms_count": sms_count,
            "data_volume_mb": data_volume_mb,
            "call_charge": round(random.uniform(100, 50000), 2),
            "cell_id": f"CELL_{random.randint(1000, 9999)}",
            "network_type": random.choice(["3G", "4G", "5G"]),
            "date_hour": start_ts.strftime("%Y-%m-%d %H:00:00")
        }
        data.append(row)

    df = pd.DataFrame(data)

    if apply_dirty:
        df = make_dirty(df, dirty_frac=0.5, seed=random.randint(1, 1000))

    return df


# ---------------- Main ----------------
if __name__ == "__main__":
    start_time = datetime(2025, 9, 20, 10, 30, 0)
    end_time = datetime(2025, 9, 20, 11, 0, 0)
    n_records = 1000

    df = generate_telecom_data(start_time, end_time, n_records, apply_dirty=True)

    output_file = os.path.join(
        output_dir,
        f"cdr_{start_time.strftime('%Y%m%d')}_{start_time.strftime('%H%M')}-{end_time.strftime('%H%M')}.csv"
    )
    df.to_csv(output_file, index=False)
    print(f"✅ File dữ liệu viễn thông (bẩn) đã được tạo với {n_records:,} bản ghi: {output_file}")
