import pandas as pd
from pathlib import Path


# Configuration
BASE_DIR = Path(r"/LANL")

AUTH_FILE = BASE_DIR / "auth.txt.gz"
FLOWS_FILE = BASE_DIR / "flows.txt.gz"
REDTEAM_FILE = BASE_DIR / "redteam.txt.gz"
DNS_FILE = BASE_DIR / "dns.txt.gz"
PROC_FILE = BASE_DIR / "proc.txt.gz"

print("BASE_DIR exists:", BASE_DIR.exists())
print("AUTH exists:", AUTH_FILE.exists(), AUTH_FILE)
print("FLOWS exists:", FLOWS_FILE.exists(), FLOWS_FILE)
print("REDTEAM exists:", REDTEAM_FILE.exists(), REDTEAM_FILE)
print("DNS exists:", DNS_FILE.exists(), DNS_FILE)
print("PROC exists:", PROC_FILE.exists(), PROC_FILE)

# Read only a sample at first so we dont overload the memory

NROWS_SAMPLE = 100000


# Column names from LANL documentation
AUTH_COLS =[
    "time",
    "src_user_domain",
    "dst_user_domain",
    "src_computer",
    "dst_computer",
    "auth_type",
    "logon_type",
    "auth_orientation",
    "success"
]


FLOWS_COLS =[
    "time",
    "duration",
    "src_computer",
    "src_port",
    "dst_computer",
    "dst_port",
    "protocol",
    "packet_count",
    "byte_count"
]

REDTEAM_COLS =[
    "time",
    "user_domain",
    "src_computer",
    "dst_computer"
]

DNS_COLS =[
    "time",
    "src_computer",
    "resolved_computer"
]

PROC_COLS =[
    "time",
    "user_domain",   # user@domain
    "computer",
    "process_name",
    "event_type"     # Start or end
]




# Helper function
def inspect_file(file_path: Path, column_names: list[str], file_label: str) -> pd.DataFrame:
    """
    Reads a small sampl from a LANL .txt.gz file and prints useful diagnostics
    - confirms the file is readable
    - confirms delimiter assumptions
    - confirms column counts match documentation
    - gives us the first look at real values

    """

    print("\n" + "=" * 80)
    print(f"INSPECTING: {file_label}")
    print(f"PATH: {file_path}")


    if not file_path.exists():
        print("ERROR: File not found.")
        return pd.DataFrame()

    # READ SAMPLE only
    compression_type = "gzip" if str(file_path).endswith(".gz") else None

    df = pd.read_csv(
        file_path,
        compression=compression_type,
        header=None,
        names=column_names,
        sep=",",
        nrows=NROWS_SAMPLE
    )

    print("\nHead:")
    print(df.head())

    print("\nShape:")
    print(df.shape)

    print("\nDtypes:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isna().sum())

    print("\nFirst 10 rows check:")
    print(df.head(10).drop_duplicates().shape[0])

    return df


if __name__ == "__main__":
    df_auth = inspect_file(AUTH_FILE, AUTH_COLS, "AUTH")
    df_flows = inspect_file(FLOWS_FILE, FLOWS_COLS, "FLOWS")
    df_red = inspect_file(REDTEAM_FILE, REDTEAM_COLS, "RED")
    df_dns = inspect_file(DNS_FILE, DNS_COLS, "DNS")
    df_proc = inspect_file(PROC_FILE, PROC_COLS, "PROC")



    # Extra sanity checks
    # --------------------------------------------------------
    if not df_auth.empty:
        print("\n" + "=" * 80)
        print("AUTH SANITY CHECKS")
        print("Unique success values:")
        print(df_auth["success"].value_counts(dropna=False).head(20))

        print("\nUnique auth types:")
        print(df_auth["auth_type"].value_counts(dropna=False).head(20))

    if not df_flows.empty:
        print("\n" + "=" * 80)
        print("FLOWS SANITY CHECKS")
        print("Unique protocols:")
        print(df_flows["protocol"].value_counts(dropna=False).head(20))

        print("\nSummary stats for numeric columns:")
        print(df_flows[["time", "duration", "packet_count", "byte_count"]].describe())

    if not df_red.empty:
        print("\n" + "=" * 80)
        print("REDTEAM SANITY CHECKS")
        print("Unique src computers:")
        print(df_red["src_computer"].nunique())

        print("Unique dst computers:")
        print(df_red["dst_computer"].nunique())

    if not df_dns.empty:
        print("\n" + "=" * 80)
        print("DNS SANITY CHECKS")
        print("Unique src domains:")
        print(df_dns["src_computer"].nunique())


    if not df_proc.empty:
        print("\n" + "=" * 80)
        print("PROC SANITY CHECKS")
        print("Unique src processes:")
        print(df_proc["src_computer"].nunique())

for f in BASE_DIR.iterdir():
    print(f.name)