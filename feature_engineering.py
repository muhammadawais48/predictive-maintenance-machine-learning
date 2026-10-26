"""Extract sensor measurements and derived features using SQLite SQL."""
import sqlite3
import pandas as pd

QUERY = """
SELECT UDI, Type, Air_temperature_K, Process_temperature_K,
       Rotational_speed_rpm, Torque_Nm, Tool_wear_min,
       (Process_temperature_K - Air_temperature_K) AS Temp_Difference,
       (Rotational_speed_rpm * Torque_Nm) AS Power_Proxy,
       Machine_failure
FROM sensor_data
ORDER BY UDI
"""

def extract_features(db_path="predictive_maintenance.db"):
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(QUERY, conn)
    if df.empty:
        raise ValueError("No sensor records found in the database")
    return df

if __name__ == "__main__":
    print(extract_features().head().to_string(index=False))
