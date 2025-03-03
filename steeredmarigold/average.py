import json
import pandas
from typing import Dict

class ValidationResult:
    def __init__(self):
        self._data = pandas.DataFrame()

    def add_value(self, sample_id:str, metrics: Dict[str, float]):
        row = {"id": [sample_id]}
        for k, v in metrics.items():
            row[k] = [v]

        row = pandas.DataFrame.from_dict(row, orient='columns')
        self._data = pandas.concat([self._data, row], axis=0, join='outer')

    def save(self, csv_path: str):
        self._data.to_csv(csv_path, index=False)

    def average_value(self, metric: str):
        return self._data[metric].mean()
    
    def average_values(self)-> Dict[str, float]:
        result = {}

        for column in self._data.columns:
            if column == "id":
                continue

            result[column] = self.average_value(column)
        
        return result

    def save_average_to_json(self, path: str):
        with open(path, "w") as file:
            json.dump({k: float(v) for k, v in self.average_values().items()}, file, indent=2, sort_keys=True)