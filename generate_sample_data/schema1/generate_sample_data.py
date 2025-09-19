import pandas as pd
import uuid
import random
from datetime import datetime, timedelta
import faker
import os

fake = faker.Faker()

output_dir = "telecom_cdr_csv"
os.makedirs(output_dir, exist_ok=True)

def generate_telecom_data(start_time, end_time, n_records):
    data = []
    for _ in range(n_records):
        start_ts = start_time + timedelta(
            seconds=random.randint(0, int((end_time - start_time).total_seconds()))
        )

        # Giả lập dịch vụ: VOICE, SMS, DATA
        service_type = random.choice(["VOICE", "SMS", "DATA"])

        duration = 0
        sms_count = 0
        data_volume_mb = 0.0

        if service_type == "VOICE":
            duration = random.randint(1, 3600)  # giây
        elif service_type == "SMS":
            sms_count = random.randint(1, 5)
        elif service_type == "DATA":
            data_volume_mb = round(random.uniform(0.1, 500.0), 2)

        end_ts = start_ts + timedelta(seconds=duration if duration > 0 else random.randint(1, 60))

        row = {
            "record_id": str(uuid.uuid4()),
            "msisdn": f"+84{random.randint(700000000, 999999999)}",
            "service_type": service_type,                 # VOICE / SMS / DATA
            "call_start": start_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "call_end": end_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_sec": duration,                     # số giây (VOICE)
            "sms_count": sms_count,                       # số SMS
            "data_volume_mb": data_volume_mb,             # dung lượng data
            "call_charge": round(random.uniform(100, 50000), 2),  # phí tính cước
            "cell_id": f"CELL_{random.randint(1000, 9999)}",
            "network_type": random.choice(["3G", "4G", "5G"]),
            "date_hour": start_ts.strftime("%Y-%m-%d %H:00:00")
        }
        data.append(row)

    return pd.DataFrame(data)


# ---------------- Main ----------------
if __name__ == "__main__":
    start_time = datetime(2025, 9, 21, 10, 0, 0)
    end_time = datetime(2025, 9, 21, 10, 30, 0)
    n_records = 1000

    df = generate_telecom_data(start_time, end_time, n_records)

    output_file = os.path.join(
        output_dir,
        f"cdr_{start_time.strftime('%Y%m%d')}_{start_time.strftime('%H%M')}-{end_time.strftime('%H%M')}.csv"
    )
    df.to_csv(output_file, index=False)
    print(f"✅ File dữ liệu viễn thông đã được tạo với {n_records:,} bản ghi: {output_file}")
