import argparse
import attr
import os
import pandas as pd
import tarfile
import urllib

@attr.s
class InputArguments:
    
    pathConfFile = attr.ib(default=None)
    
    @classmethod
    def parse_args(cls, parser: argparse.ArgumentParser=None):
        
        if parser is None:
            parser = argparse.ArgumentParser()
            
        parser.add_argument(
            "-pathConfFile", '--config', help="Path where configuration files are contained"
        )
        
        args = parser.parse_args()
        
        return cls(pathConfFile=args.pathConfFile)


class DataAccessor:
    
    def __init__(self, **kwargs):
        """
        Access housing data.
        """
        self.housing_url = kwargs.get("HOUSING_URL")
        self.housing_path = kwargs.get("INPUT_PATH")
        self.housing_filename = kwargs.get('INPUT_FILENAME', "housing.csv")
        self.fetch_data = eval(kwargs.get("FETCH_DATA", True))
        
    def fetch_housing_data(self):
        """
        Download housing data from url in a datasets/housing folder.
        """
        # create datasets directory if it doesn't exists
        if not os.path.isdir(self.housing_path):
            os.makedirs(self.housing_path)
        tgz_path = os.path.join(self.housing_path, "housing.tgz")

        # download housing data in the path defined above
        urllib.request.urlretrieve(self.housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=self.housing_path)
        housing_tgz.close()
    
    def load_housing_data(self):
        """
        Upload data from datasets/housing folder.
        """
        if self.fetch_data:
            self.fetch_housing_data()
        csv_path = os.path.join(self.housing_path, self.housing_filename)
        print("Housing data loaded successfully.")
        return pd.read_csv(csv_path)