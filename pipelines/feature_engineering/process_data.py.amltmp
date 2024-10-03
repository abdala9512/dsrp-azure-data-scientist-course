import pandas as pd
import numpy as np
import argparse
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# CREDENCIALES AZURE
# authenticate
credential = DefaultAzureCredential()

SUBSCRIPTION="0f61d6bf-ab3d-4df7-a666-edaf42eff57c"
RESOURCE_GROUP="AML-COURSE-DP100-2024"
WS_NAME="dsrp-aml-dp100"
# Get a handle to the workspace
ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WS_NAME,
)# METADATA ASSET


parser = argparse.ArgumentParser()
parser.add_argument("--input_data_version", type=int, default=1)
args = parser.parse_args()

INPUT_DATA_ASSET = "booking-dsrp"
data_asset = ml_client.data.get(INPUT_DATA_ASSET, version=str(args.input_data_version))

dataframe_reservas_hotel_raw = pd.read_csv(data_asset.path)

from typing import List, Union

class DataProcessor:
    """
    Procesador de datos de Reservas de hoteles
    """

    def __init__(self, client: MLClient,  data: pd.DataFrame):
        self.client = client
        self.data = data


    def __encode_with_ohe(self, col_names: Union[str, List[str]]):
        """
        ONE HOT ENCODING DE CUALQUIER COLUMNA

        col_names: LIST DE COLUMNAS PARA APLICAR OHE
        """
        encoded_dfs = []
        for col in col_names:

            encoder = OneHotEncoder()
            fitted_encoder = encoder.fit(self.data[[col]])
            encoded_array = fitted_encoder.transform(self.data[[col]]).toarray()
            encoded_df = pd.DataFrame(encoded_array, columns= fitted_encoder.get_feature_names_out())
            encoded_dfs.append(encoded_df)

        return pd.concat(encoded_dfs, axis=1)


    def process(self):

        """
        PROCESAMIENTO PRINCIPAL
        """

        # VARIABLE QUE CALCULA SI EL HOTEL ESTA EN LA CIUDAD
        self.data["is_city_hotel"] = self.data["hotel"].apply(lambda x: int(x == "City Hotel"))
        self.data = self.data.drop("hotel", axis=1 )

         # CREAR DIFERENCIA ENTRE CUARTO RESERVADO Y CUARTO ASIGNADO
        self.data["diff_room"] = np.where(self.data["assigned_room_type"] == self.data["reserved_room_type"], 1, 0)


        # LISTA DE COLUMNAS PARA HACER ONE HOT ENCODDING 
        ohe_column_list = [
            "arrival_date_year", 
            "arrival_date_month",
            "meal",
            "market_segment",
            "distribution_channel", 
            "reserved_room_type",
            "deposit_type",
            "customer_type"
        ]

        encoded_dfs = self.__encode_with_ohe(col_names=ohe_column_list)
        self.data = pd.concat([self.data, encoded_dfs], axis=1)
        self.data = self.data.drop(ohe_column_list, axis=1)

        # IMPUTACION VARIABLE CHILDREN
        self.data["children"] = dataframe_reservas_hotel_raw["children"].fillna(0)

       

        # COLUMNAS QUE VAMOS A ELIMINAR
        drop_list = [
            "country", 
            "company", 
            "agent",
            "reservation_status", 
            "reservation_status_date",
            "name", 
            "email", 
            "phone-number", 
            "credit_card",
            "assigned_room_type"
        ]

        self.data = self.data.drop(drop_list, axis=1)

        # FEATURE SELECTOR - VARIANZA
        selector = VarianceThreshold()
        selected_variables = selector.fit_transform(self.data)

        print([ i for i in self.data.columns  if i not in selector.get_feature_names_out() ] )
        self.data = pd.DataFrame(selected_variables, columns=selector.get_feature_names_out())

        # STANDARDIZATION
        std_vars = ["lead_time", "adr", "days_in_waiting_list"]
        scales = []
        for col in std_vars:
            scaler = StandardScaler()  # OTROS METODOS MinMaxScaler, RobustScaler
            scaled_var = pd.DataFrame(
                scaler.fit_transform(dataframe_reservas_hotel_raw[col].to_numpy().reshape(-1, 1)), columns=[col]
            )
            scales.append(scaled_var)
        
        scales = pd.concat(scales, axis=1)
        self.data = self.data.drop(std_vars, axis=1)
        self.data = pd.concat([self.data, scales], axis=1)

    def write(self):

        GOLD_DATA_ASSET = "gold-booking-dsrp"
        latest_version = int(
            [
                asset.latest_version for asset in ml_client.data.list() if asset.name == GOLD_DATA_ASSET
            ][0]
        )
        PROCESSED_DATA_PATH = "feature_engineering_data_PROCESSED.csv"
        self.data.to_csv(PROCESSED_DATA_PATH,index=False)
        processed_dataset = Data(
            path=PROCESSED_DATA_PATH,
            type=AssetTypes.URI_FILE,
            description="Tabla Final Feature Engineering",
            name=GOLD_DATA_ASSET,
            version=str(latest_version + 1)
        )
        self.client.data.create_or_update(processed_dataset)


# EJECUCION PIPELINE
processor = DataProcessor(client=ml_client, data=dataframe_reservas_hotel_raw)
processor.process()
processor.write()

