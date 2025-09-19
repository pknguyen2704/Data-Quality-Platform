import pandas as pd
import uuid
import random
from datetime import datetime, timedelta
import faker
import os
import numpy as np

fake = faker.Faker()

output_dir = "telecom_usage_csv"
os.makedirs(output_dir, exist_ok=True)


# ---------------- Hàm làm bẩn ----------------
def make_dirty(df_pd, dirty_frac=0.3, seed=42, exclude_cols=None):
    rng = np.random.default_rng(seed)
    df = df_pd.copy()
    n_dirty = int(len(df) * dirty_frac)
    if n_dirty == 0:
        return df

    available_features = [c for c in df.columns if c not in (exclude_cols or [])]
    if not available_features:
        return df

    k = min(5, len(available_features))
    features_to_dirty = rng.choice(available_features, size=k, replace=False)

    for col in features_to_dirty:
        idx = rng.choice(df.index, size=n_dirty, replace=False)

        for row_idx in idx:
            if rng.random() < 0.5:
                df.at[row_idx, col] = None
            else:
                if pd.api.types.is_object_dtype(df[col]):
                    val = str(df.at[row_idx, col])
                    dirty_suffix = "_" + "".join(rng.choice(list("ABCXYZ0123456789"), size=5))
                    df.at[row_idx, col] = val + dirty_suffix
                elif pd.api.types.is_numeric_dtype(df[col]):
                    val = df.at[row_idx, col]
                    if val is not None:
                        df.at[row_idx, col] = val + rng.integers(100, 1000)

    return df


# ---------------- Hàm sinh dữ liệu ----------------
def generate_telecom_usage(start_time, end_time, n_records, apply_dirty=False):
    data = []
    for _ in range(n_records):
        start_ts = start_time + timedelta(
            seconds=random.randint(0, int((end_time - start_time).total_seconds()))
        )

        service_type = random.choice(["VOICE", "SMS", "DATA"])
        duration = 0
        sms_count = 0
        volume_mb = 0.0

        if service_type == "VOICE":
            duration = random.randint(10, 3600)
        elif service_type == "SMS":
            sms_count = random.randint(1, 5)
        elif service_type == "DATA":
            volume_mb = round(random.uniform(0.1, 500.0), 2)

        end_ts = start_ts + timedelta(seconds=duration if duration > 0 else random.randint(1, 60))
        charge_amount = round(random.uniform(100, 50000), 2)

        row = {
            "session_id": str(uuid.uuid4()),
            "msisdn": f"+84{random.randint(700000000, 999999999)}",
            "service_type": service_type,
            "volume_mb": volume_mb,
            "sms_count": sms_count,
            "duration_sec": duration,
            "charge_amount": charge_amount,
            "cell_id": f"CELL_{random.randint(1000, 9999)}",
            "network_type": random.choice(["3G", "4G", "5G"]),
            "event_start": start_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "event_end": end_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "date_hour": start_ts.strftime("%Y-%m-%d %H:00:00")  # giữ nguyên
        }
        data.append(row)

    df = pd.DataFrame(data)

    if apply_dirty:
        df = make_dirty(df, dirty_frac=0.3, seed=random.randint(1, 1000), exclude_cols=["date_hour"])

    return df


# ---------------- Main ----------------
if __name__ == "__main__":
    start_time = datetime(2025, 9, 21, 10, 0, 0)
    end_time = datetime(2025, 9, 21, 10, 30, 0)
    n_records = 1000

    # True = dữ liệu bẩn, False = dữ liệu sạch
    df = generate_telecom_usage(start_time, end_time, n_records, apply_dirty=True)

    output_file = os.path.join(
        output_dir,
        f"usage_{start_time.strftime('%Y%m%d')}_{start_time.strftime('%H%M')}-{end_time.strftime('%H%M')}.csv"
    )
    df.to_csv(output_file, index=False)
    print(f"✅ File dữ liệu telecom usage đã được tạo với {n_records:,} bản ghi: {output_file}")
