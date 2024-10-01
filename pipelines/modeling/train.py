import pandas as pd
import mlflow


from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report, 
    roc_auc_score, 
    confusion_matrix
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from hyperopt import fmin, hp, tpe, Trials
from loguru import logger
from typing import Any



# CREDENCIALES AZURE
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
)
# METADATA ASSET
data_asset = ml_client.data.get("gold-booking-dsrp", version="2")

modeling_dataframe = pd.read_csv(data_asset.path)
TARGET_COLUMN = "is_canceled"

X = modeling_dataframe.drop(TARGET_COLUMN, axis=1)
y = modeling_dataframe[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)


class MachineLearningProcessor:

    def __init__(self, 
        data: pd.DataFrame, 
        algorithm: any, 
        model_name: str, 
        target:str, 
        params: dict = None):

        self.data = data
        self.algorithm = algorithm
        self.model_name = model_name
        self.target_column = target

    def _split_data(self):

        X = self.data.drop(self.target_column, axis=1)
        y = self.data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)

        return X_train, X_test, y_train, y_test

    def __make_pipeline(self, params: dict = None) -> Pipeline:

        if params:
            algorithm = self.algorithm(**params)
        else:
            algorithm = self.algorithm()

        _pipeline = Pipeline(steps=[
            ("std_scaling", StandardScaler()),
            ("classifier",algorithm )
            ]
        )

        return _pipeline

    def optimize_grid_search(self, search_space: dict):

        X_train, X_test, y_train, y_test = self._split_data()
        _pipeline = self.__make_pipeline()
        
        optimizer = GridSearchCV(
            _pipeline,
            param_grid={
                f"classifier__{param}": space 
                for param, space in search_space.items()
            },
            cv=3
        )
        optimizer.fit(X_train, y_train)
        return optimizer.cv_results_

    def optimize_random_search(self, search_space: dict):

        X_train, X_test, y_train, y_test = self._split_data()
        _pipeline = self.__make_pipeline()
        
        optimizer = RandomizedSearchCV(
            _pipeline,
            param_distributions={
                f"classifier__{param}": space 
                for param, space in search_space.items()
            },
            random_state=100,
            n_iter=5,
            cv=3
        )
        optimizer.fit(X_train, y_train)
        return optimizer.best_params_

    def optimize_tpe(self, search_space:dict):

        def objective(params):
            """
            Entrenar modelo y devolver metrica ML
            """
            X_train, X_test, y_train, y_test = self._split_data()
            
            _pipeline = self.__make_pipeline(params=params)

            _pipeline.fit(X_train, y_train)
            predictions = _pipeline.predict(X_test)

            return -accuracy_score(y_test, predictions)

        trials =Trials()
        best = fmin(
            fn=objective,
            space=search_space,
            max_evals=10,
            algo=tpe.suggest
        )
        return best
        
        

    def train(self, params: dict):
        """
        Entrenamiento del model de ML
        """
        mlflow.autolog()
        with mlflow.start_run(run_name=self.model_name):

            X_train, X_test, y_train, y_test = self._split_data()

            _pipeline = self.__make_pipeline(params=params)

            _pipeline.fit(X_train, y_train)
            predictions = _pipeline.predict(X_test)

            metrics = {
                "accuracy_score": accuracy_score(y_test, predictions),
                "recall_score": recall_score(y_test, predictions),
                "precision_score":precision_score(y_test, predictions),
                "f1_score": f1_score(y_test, predictions)
            }

            logger.info(f" {self.model_name} Accuracy: {metrics['accuracy_score']}")
            logger.info(f" {self.model_name} Recall: {metrics['recall_score']}")
            logger.info(f" {self.model_name} Precision: {metrics['precision_score']}")
            logger.info(f" {self.model_name} F1-Score: {metrics['f1_score']}")

            print(confusion_matrix(y_test, predictions))
            print(
                classification_report(y_test, predictions)
            )

xgboost_ml_processor = MachineLearningProcessor(
    data=modeling_dataframe,
    algorithm=XGBClassifier,
    model_name="JOB - XGBoost Classifier with MachineLearningProcessor",
    target=TARGET_COLUMN,
)
best_params = xgboost_ml_processor.optimize_tpe(
    search_space={
        "max_depth": hp.randint("max_depth", 1, 10),
    }
)
xgboost_ml_processor.train(
    params=best_params
)
