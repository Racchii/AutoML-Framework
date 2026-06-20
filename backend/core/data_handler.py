import pandas as pd
from fastapi import UploadFile
import io

class DataHandler:
    @staticmethod
    async def read_csv_file(file: UploadFile) -> pd.DataFrame:
        content = await file.read()
        # Assume it's a CSV for now
        df = pd.read_csv(io.BytesIO(content))
        return df

    @staticmethod
    def infer_schema(df: pd.DataFrame) -> dict:
        schema = {
            "columns": df.columns.tolist(),
            "types": df.dtypes.astype(str).to_dict(),
            "shape": df.shape,
            "missing_values": df.isnull().sum().to_dict()
        }
        return schema
