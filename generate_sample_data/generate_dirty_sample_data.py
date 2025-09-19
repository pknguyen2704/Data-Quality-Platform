import pandas as pd
import uuid
import random
from datetime import datetime, timedelta
import faker
import os
import numpy as np
import json

fake = faker.Faker()

output_dir = "dirty_mysql_csv"
os.makedirs(output_dir, exist_ok=True)

time_range = (datetime(2025, 9, 10, 10, 30, 0), datetime(2025, 9, 10, 11, 0, 0))
n_records = 10


EXCLUDE_COLUMNS = [
    "call_uid","call_start_time","call_response_time","call_ringing_time",
    "call_answer_time","call_end_time","call_date","call_hour",
    "session_uid","start_ts_epoch","response_ts_epoch","ringing_ts_epoch",
    "answer_ts_epoch","end_ts_epoch"
]

# ---------------- Hàm làm bẩn ----------------
def make_dirty(df_pd, dirty_frac=0.5, unuse_columns=None, seed=42):
    if unuse_columns is None:
        unuse_columns = []
    rng = np.random.default_rng(seed)
    df = df_pd.copy()
    n_dirty = int(len(df) * dirty_frac)
    if n_dirty == 0:
        return df

    available_features = [c for c in df.columns if c not in unuse_columns]
    if not available_features:
        return df
    k = min(8, len(available_features))
    features_to_dirty = rng.choice(available_features, size=k, replace=False)

    for col in features_to_dirty:
        idx = rng.choice(df.index, size=n_dirty, replace=False)
        mask = rng.random(n_dirty) < 0.5

        if pd.api.types.is_numeric_dtype(df[col]):
            df.loc[idx[mask], col] = None
            df.loc[idx[~mask], col] = df[col].median() * rng.uniform(10, 50)

        elif pd.api.types.is_object_dtype(df[col]):
            for i, row_idx in enumerate(idx):
                if mask[i]:
                    df.at[row_idx, col] = "CORRUPTED"
                else:
                    val = str(df.at[row_idx, col])
                    corrupted_val = "".join(rng.choice(list("ABCXYZ012345"), len(val)))
                    df.at[row_idx, col] = corrupted_val

    return df

# ---------------- Hàm sinh dữ liệu ----------------
def generate_data(start_time, end_time, n_records):
    data = []
    for i in range(n_records):
        start_ts = start_time + timedelta(
            seconds=random.randint(0, int((end_time - start_time).total_seconds()))
        )

        response_delta = timedelta(seconds=random.randint(1, 5))
        ringing_delta = timedelta(seconds=random.randint(1, 5))
        answer_delta = timedelta(seconds=random.randint(1, 10))
        end_delta = timedelta(seconds=random.randint(30, 3600))

        row = {
            "to_user": fake.user_name(),
            "from_user": fake.user_name(),
            "call_uid": str(uuid.uuid4()),
            "call_start_time": start_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "call_response_time": (start_ts + response_delta).strftime("%Y-%m-%d %H:%M:%S"),
            "call_ringing_time": (start_ts + ringing_delta).strftime("%Y-%m-%d %H:%M:%S"),
            "call_answer_time": (start_ts + answer_delta).strftime("%Y-%m-%d %H:%M:%S"),
            "call_end_time": (start_ts + end_delta).strftime("%Y-%m-%d %H:%M:%S"),
            "end_method": random.choice(["BYE", "CANCEL"]),
            "end_reason": random.choice(["NORMAL", "BUSY", "NO_ANSWER"]),
            "last_sip_method": random.choice(["INVITE", "UPDATE", "ACK"]),
            "to_tag_val": json.dumps([str(uuid.uuid4()) for _ in range(random.randint(1, 3))]),
            "from_tag_val": str(uuid.uuid4()),
            "session_uid": str(uuid.uuid4()),
            "early_media_mode": random.choice(["sendrecv", "sendonly", "recvonly", "inactive", "gated"]),
            "caller_user_agent": fake.user_agent(),
            "mo_network_info": fake.ipv4(),
            "mt_network_info": fake.ipv4(),
            "caller_identity_enc": str(uuid.uuid4()),
            "alert_info": random.choice([None, "WARNING_1", "WARNING_2"]),
            "src_ip": fake.ipv4(),
            "dst_ip": fake.ipv4(),
            "mo_contact_uri": fake.uri(),
            "mt_contact_uri": fake.uri(),
            "to_phone_masked": str(random.randint(84000000000, 84999999999)),
            "from_phone_masked": str(random.randint(84000000000, 84999999999)),
            "start_ts_epoch": int(start_ts.timestamp()),
            "response_ts_epoch": int((start_ts + response_delta).timestamp()),
            "ringing_ts_epoch": int((start_ts + ringing_delta).timestamp()),
            "answer_ts_epoch": int((start_ts + answer_delta).timestamp()),
            "end_ts_epoch": int((start_ts + end_delta).timestamp()),
            "to_enc_local": random.randint(1000, 9999),
            "from_enc_public": fake.uri(),
            "to_phone_enc": str(random.randint(84000000000, 84999999999)),
            "from_phone_enc": str(random.randint(84000000000, 84999999999)),
            "caller_identity_enc_mask": str(uuid.uuid4()),
            "sdp_req_ip": fake.ipv4(),
            "sdp_req_port": random.randint(1000, 65535),
            "sdp_req_media": random.choice(["audio", "video", "audio/video"]),
            "sdp_resp_ip": fake.ipv4(),
            "sdp_resp_port": random.randint(1000, 65535),
            "sdp_resp_media": random.choice(["audio", "video", "audio/video"]),
            "call_status": random.choice(["DROP", "FAIL", "SUCCESS"]),
            "call_type": random.choice(["MO", "MT"]),
            "terminated_by": random.choice(["MO", "MT"]),
            "sip_route": fake.uri(),
            "pcap_files_list": json.dumps([f"file_{i}.pcap" for i in range(random.randint(1, 3))]),
            "handler_info": fake.user_name(),
            "mt_user_agent_info": fake.user_agent(),
            "end_method_1": random.choice(["BYE", "CANCEL"]),
            "end_reason_1": random.choice(["NORMAL", "BUSY", "NO_ANSWER"]),
            "terminated_by_1": random.choice(["MO", "MT"]),
            "terminate_src_ip": fake.ipv4(),
            "terminate_src_ip_1": fake.ipv4(),
            "dpi_node_ip": fake.ipv4(),
            "call_date": start_ts.strftime("%Y-%m-%d"),
            "call_hour": start_ts.strftime("%H:00:00")
        }
        data.append(row)
    df = pd.DataFrame(data)

    # Làm bẩn dữ liệu
    df_dirty = make_dirty(df, dirty_frac=0.5, unuse_columns=EXCLUDE_COLUMNS, seed=random.randint(1, 1000))
    return df_dirty

# ---------------- Main ----------------
start, end = time_range
df_dirty = generate_data(start, end, n_records)

output_file = os.path.join(
    output_dir,
    f"mysql_dirty_{start.strftime('%Y%m%d')}_{start.strftime('%H%M')}-{end.strftime('%H%M')}.csv"
)
df_dirty.to_csv(output_file, index=False)
print(f"✅ File dữ liệu cực bẩn (MySQL compatible) đã được tạo với {n_records:,} bản ghi: {output_file}")
