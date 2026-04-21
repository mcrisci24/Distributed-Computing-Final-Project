# ----PySpark Pipeline-----
# --Read raw LANL files from S3
# --Apply schema
# --Clean and standardize each source
# --Write silver outputs as Delta


BUCKET = "lanl-cyber-pipeline"
RAW_BASE = f"s3://{BUCKET}/lanl/raw"
SILVER_BASE = f"s3://{BUCKET}/lanl/silver"

AUTH_PATH = f"s3://{RAW_BASE}/auth.txt.gz"
FLOWS_PATH = f"s3://{RAW_BASE}/flows.txt.gz"
DNS_PATH = f"s3://{RAW_BASE}/dns.txt.gz"
PROC_PATH = f"s3://{RAW_BASE}/proc.txt.gz"
REDTEAM_PATH = f"s3://{RAW_BASE}/redteam.txt.gz"

DEBUG_SAMPLE = True
DEBUG_SAMPLE_FRACTION = 0.2


print("=" * 90)
print("LANL SILVER Pipeline STARTED")
print(f"RAW_BASE: {RAW_BASE}")
print(f"SILVER_BASE: {SILVER_BASE}")
print(f"DEBUG_SAMPLE: {DEBUG_SAMPLE}")
print("=" * 90)


# Schemas
auth_schema = StructType([
    StructField("time", LongType(), True),
    StructField("src_user_domain", StringType(), True),
    StructField("dst_user_domain", StringType(), True),
    StructField("src_computer", StringType(), True),
    StructField("dst_computer", StringType(), True),
    StructField("auth_type", StringType(), True),
    StructField("logon_type", StringType(), True),
    StructField("auth_orientation", StringType(), True),
    StructField("success", StringType(), True),
])


flows_schema = StructType([
    StructField("time", LongType(), True),
    StructField("duration", LongType(), True),
    StructField("src_computer", StringType(), True),
    StructField("src_port", StringType(), True),
    StructField("dst_computer", StringType(), True),
    StructField("dst_port", StringType(), True),
    StructField("protocol", IntegerType(), True),
    StructField("packet_count", LongType(), True),
    StructField("byte_count", LongType(), True),

])


dns_schema = StructType([
    StructField("time", LongType(), True),
    StructField("src_computer", StringType(), True),
    StructField("resolved_computer", StringType(), True),
])



proc_schema = StructType([
    StructField("time", LongType(), True),
    StructField("user_domain", StringType(), True),
    StructField("src_computer", StringType(), True),
    StructField("dst_computer", StringType(), True),
])


redteam_schema = StructType([
    StructField("time", LongType(), True),
    StructField("user_domain", StringType(), True),
    StructField("src_computer", StringType(), True),
    StructField("dst_computer", StringType(), True),

])



# Helper Functions
def debug_df(df, name: str, show_n: int = 5) -> None:
    """
    Structured debugging function to check spark dataframes
    """
    print("\n" + "=" * 90)
    print(f"Debug CHECKING: {name}")
    print("=" *90)
    print("Columns:", df.columns)
    print("Schema:")
    df.printSchema()
    row_count = df.count()
    print(f"Rows: {row_count}")
    print("Preview:")
    df.show(show_n, truncate=False)


def sample(df):
    """
    Debug sampling to reduce runtime
    """
    if DEBUG_SAMPLE:
        return df.sample(withReplacement=False, fraction=DEBUG_SAMPLE_FRACTION, seed =123)
    return df


def read_raw_csv(path: str, schema: StructType, label: str):
    """
    Read one raw LANL source from S3 with schema above appleid
    """
    print("\n" + "=" * 90)
    print("READING Raw Source:", {label})
    print(f"PATH: {path}")

    df = (
        spark.read
        .option("header", False)
        .option("sep", ",")
        .schema(schema)
        .csv(path)
    )

    df = sample(df)
    debug_df(df, f"{label} RAW")
    return df


def trim_string_columns(df):
    """"
    Trim all string columns
    """

    for field in df.schema.fields:
        if field.dataType.simpleString() == "string":
            df = df.withColumn(field.name, F.trim(F.col(field.name)))

    return df


def basic_cleaning(df, label:str):
    """
    Cleaning:
    --- Trim string columns
    --- cast time to bigint if present
    --- add data quality checks
    """

    print("\n" + "-" * 90)
    print(f"CLEANING SOURCE: {label}")

    df  = trim_string_columns(df)

    if "time" in df.columns:
        df = df.withColumn("time", F.col("time").cast("bigint"))


    # Debugs null counts
    null_exprs =[
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in df.columns
    ]
    print(f"Null counts for {label}:")
    df.select(null_exprs).show(truncate=False)

    debug_df(df, f"{label} Cleaned")
    return df


def save_delta(df, path:str, label:str):
    """
    Silver output as Delta
    """
    print("\n" + "-" * 90)
    print(f"WRITING DELTA: {label}")
    print(f"DESTINATION: {path}")

    (
        df.write
        .format("delta")
        .mode("overwrite")
        .save(path)
    )

    print(f"Saved {label} to Delta successfully.")



# ------ AUTH Pipeline ------
auth_df = read_raw_csv(AUTH_PATH, auth_schema, "AUTH")
auth_df = basic_cleaning(auth_df, "AUTH")

auth_df = auth_df.withColumn(
    "success_flag",
    F.when(F.upper(F.col("success")) == "SUCCESS", F.lit(1)).otherwise(F.lit(0))
)

print("AUTH unique src computers:")
auth_df.select(F.countDistinct("src_computer").alias("unique_src_computers")).show()

print("AUTH unique dst computers:")
auth_df.select(F.countDistinct("dst_computer").alias("unique_dst_computers")).show()


save_delta(auth_df, AUTH_OUT, "AUTH")



# FLOWS Pipeline
flows_df = read_raw_csv(FLOWS_PATH, flows_schema, "FLOWS")
flows_df = basic_clean(flows_df, "FLOWS")

flows_df = (
    flows_df
    .withColumn("duration", F.col("duration").cast("bigint"))
    .withColumn("protocol", F.col("protocol").cast("int"))
    .withColumn("packet_count", F.col("packet_count").cast("bigint"))
    .withColumn("byte_count", F.col("byte_count").cast("bigint"))
)

print("FLOWS unique src computers:")
flows_df.select(F.countDistinct("src_computer").alias("unique_flows")).show()

print("FLOWS unique dst computers:")
flows_df.select(F.countDistinct("dst_computer").alias("unique_flows")).show()

save_delta(flows, FLOWS_OUT, "FLOWS")



# DNS Pipeline

dns_df = read_raw_csv(DNS_PATH, dns_schema, "DNS")
dns_df = basic_cleaning(dns_df, "DNS")

print("DNS unique src computers:")
dns_df.select(F.countDistinct("src_computer").alias("unique_dns")).show()

print("DNS unique dst computers:")
dns_df.select(F.countDistinct("dst_computer").alias("unique_dns")).show()

save_delta(dns_df, DNS_OUT, "DNS")



# PROC Pipeline
proc_df = read_raw_csv(PROC_PATH, proc_schema, "PROC")
proc_df = basic_cleaning(proc_df, "PROC")

proc_df = proc_df.withColumn("event_type", F.upper(F.col("event_type")))

print("PROC unique computers:")
proc_df.select(F.countDistrinct("computer").alias("unique_computers")).show()

print("PROC event type distribution:")
proc_df.groupBy("event_type").count().orderBy(F.desc("count")).show(truncate=False)

save_delta(proc_df, PROC_OUT, "PROC")


# REDTEAM Pipeline
redteam_df = read_raw_csv(REDTEAM_PATH, redteam_schema, "REDTEAM")
redteam_df = basic_clean(redteam_df, "REDTEAM")

redteam_df = redteam_df.withColumn("redteam_flag", F.lit(1))

print("REDTEAM unique src computers:")
redteam_df.select(F.countDistinct("src_computer").alias("unique_src_computers")).show()

print("REDTEAM unique dst computers:")
redteam_df.select(F.countDistinct("dst_computer").alias("unique_dst_computers")).show()

save_delta(redteam_df, REDTEAM_OUT, "REDTEAM")


# Final checks
print("\n" + "=" * 90)
print("FINAL SILVER DELTA CHECKS")
print("=" * 90)

for label, path in [
    ("AUTH", AUTH_OUT),
    ("FLOWS", FLOWS_OUT),
    ("DNS", DNS_OUT),
    ("PROC", PROC_OUT),
    ("REDTEAM", REDTEAM_OUT),
]:
    print(f"\nReading back Delta table for {label}: {path}")
    check_df = spark.read.format("delta").load(path)
    print(f"{label} row count: {check_df.count():,}")
    check_df.printSchema()

print("\nLANL SILVER PIPELINE COMPLETE.")


