import pandas as pd
import uuid
import random
from datetime import datetime, timedelta
import faker
import os

fake = faker.Faker()

output_dir = "telecom_usage_csv"
os.makedirs(output_dir, exist_ok=True)

def generate_telecom_usage(start_time, end_time, n_records):
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
            duration = random.randint(10, 3600)  # giây
        elif service_type == "SMS":
            sms_count = random.randint(1, 5)
        elif service_type == "DATA":
            volume_mb = round(random.uniform(0.1, 500.0), 2)  # MB

        end_ts = start_ts + timedelta(seconds=duration if duration > 0 else random.randint(1, 60))
        charge_amount = round(random.uniform(100, 50000), 2)  # VNĐ

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
            "date_hour": start_ts.strftime("%Y-%m-%d %H:00:00")
        }
        data.append(row)

    return pd.DataFrame(data)


# ---------------- Main ----------------
if __name__ == "__main__":
    start_time = datetime(2025, 9, 20, 10, 0, 0)
    end_time = datetime(2025, 9, 21, 10, 30, 0)
    n_records = 1000

    df = generate_telecom_usage(start_time, end_time, n_records)

    output_file = os.path.join(
        output_dir,
        f"usage_{start_time.strftime('%Y%m%d')}_{start_time.strftime('%H%M')}-{end_time.strftime('%H%M')}.csv"
    )
    df.to_csv(output_file, index=False)
    print(f"✅ File dữ liệu viễn thông (usage records) đã được tạo với {n_records:,} bản ghi: {output_file}")
