# Proyecto Azure Machine Learning Exámen DP100

El proyecto trabajado fue un caso de uso de clasificación de cancelación de reservas de hotel

## Datos utilizados 

Tomado de Datasets de Kaggle: https://www.kaggle.com/code/touba7/hotel-booking

## Arquitectura trabajada

![image](https://github.com/user-attachments/assets/9cc27657-bd9e-4622-9a97-8659ccd844ad)

## End to End Caso de uso

### configuración previa

1. Creación Azure Machine Learning Workspace (Clase 1)
![image](https://github.com/user-attachments/assets/60d0f6b8-8b51-4338-852e-65def56c3df6)
2. Conexión con repositorio remoto de Github: https://github.com/abdala9512/dsrp-azure-data-scientist-course/blob/main/azure-ml/common/config_gh_ssh.sh
  - Configuración llaves SSH con github: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent
3. Creación Kernels como ambientes de trabajo en AML: https://github.com/abdala9512/dsrp-azure-data-scientist-course/blob/main/azure-ml/common/config_kernel.sh

#### RECOMENDADO - Creación ambientes en AML

Se recomienda crear ambiente de Azure Machine Learning para distintas cargas de trabajo, documentación sobre ambientes acá: https://learn.microsoft.com/es-es/azure/machine-learning/concept-environments?view=azureml-api-2

### Extracción de datos

[Documentación Azure para ingesta de datos](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-datastore?view=azureml-api-2&tabs=sdk-identity-based-access%2Csdk-adls-identity-access%2Csdk-azfiles-accountkey%2Csdk-adlsgen1-identity-access%2Csdk-onelake-identity-access)

Proceso de ingesta de datos ejecutado en este notebook: https://github.com/abdala9512/dsrp-azure-data-scientist-course/blob/main/azure-ml/notebooks/ingesta_datos.ipynb

### Feature Engineering
1. Notebook de trabajo Feature Engineering: https://github.com/abdala9512/dsrp-azure-data-scientist-course/blob/main/azure-ml/notebooks/preparacion_datos3.ipynb
2. Operacionalización Feature Engineering con Jobs
  - Jobs: https://github.com/abdala9512/dsrp-azure-data-scientist-course/blob/main/azure-ml/notebooks/JOBS.ipynb
  - Código fuente Feature Engineering: https://github.com/abdala9512/dsrp-azure-data-scientist-course/blob/main/azure-ml/pipelines/feature_engineering/process_data.py

### Modelamiento
1. Notebook de trabajo pipelines de Machine Learning: https://github.com/abdala9512/dsrp-azure-data-scientist-course/blob/main/azure-ml/notebooks/modelamiento.ipynb
2. Operacionalización Machine Learning con Jobs
  - Jobs: https://github.com/abdala9512/dsrp-azure-data-scientist-course/blob/main/azure-ml/notebooks/JOBS.ipynb
  - Código fuente Machine Learning: https://github.com/abdala9512/dsrp-azure-data-scientist-course/blob/main/azure-ml/pipelines/modeling/train.py
### Despliegue

Notebook Deployment: https://github.com/abdala9512/dsrp-azure-data-scientist-course/blob/main/azure-ml/notebooks/predicciones.ipynb

#### Solución 1: Batch Deployment
![image](https://github.com/user-attachments/assets/af392082-c481-4907-b5f4-219273ab06e7)

#### Solución 2: Online Deployment
![image](https://github.com/user-attachments/assets/0db1e1bc-c711-4f78-a0dc-64a7919c6a1d)

### Monitoreo
![image](https://github.com/user-attachments/assets/6b70785a-8512-454e-a9b8-33b4280889d5)

Notebook Monitoreo:https://github.com/abdala9512/dsrp-azure-data-scientist-course/blob/main/azure-ml/notebooks/monitoreo.ipynb
Monitores implementados:
- Model Performance
- DataDrift
- Sistema de Alertas por correo
Documentación Azure ML - Monitoreo: https://learn.microsoft.com/en-us/azure/machine-learning/concept-model-monitoring?view=azureml-api-2

