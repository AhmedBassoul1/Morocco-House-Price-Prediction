import os
import zipfile
from abc import ABC, abstractmethod
import pandas as pd


class Data_Ingestor(ABC):
    
    @abstractmethod
    def ingest(self, source: str) -> pd.DataFrame:
        pass


class Zip_Data_Ingestor(Data_Ingestor):
    
    def ingest(self, source: str) -> pd.DataFrame:
        if not zipfile.is_zipfile(source):
            raise ValueError(f"{source} is not a valid zip file.")
        
        with zipfile.ZipFile(source, 'r') as zip_ref:
            zip_ref.extractall("extracted_data")
        
        data_frames = []
        for file in os.listdir("extracted_data"):
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join("extracted_data", file))
                data_frames.append(df)
        
        return pd.concat(data_frames, ignore_index=True)


class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> Data_Ingestor:
        if file_extension == ".zip":
            return Zip_Data_Ingestor()
        else:
            raise ValueError(f"No ingestor available for file extension: {file_extension}")
        
if __name__ == "__main__":
    # Specify the file path
    file_path = "data/housing_data.zip"

    # Determine the file extension
    file_extension = os.path.splitext(file_path)[1]
    print(file_extension)

    # Get the appropriate DataIngestor
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    # Ingest the data and load it into a DataFrame
    df = data_ingestor.ingest(file_path)

    # Now df contains the DataFrame from the extracted CSV
    print(df.head())  # Display the first few rows of the DataFrame
