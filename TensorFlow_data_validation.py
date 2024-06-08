from __future__ import print_function
import pandas as pd
import tensorflow as tf
import tensorflow_data_validation as tfdv

#importando dataset
dataset=pd.read_csv("pollution-small 1.csv")
print(dataset.head())

training_data = dataset[:1600] #banco treino
print(training_data.describe())

test_set = dataset[1600:] #banco de teste
print(test_set.describe())

# Análise de dados e validação com TFDV
train_stats = tfdv.generate_statistics_from_dataframe(dataframe = training_data)
print(train_stats)

#inferindo o esquema (estrutura) mostra se o parametro é obrigatorio(required)
schema= tfdv.infer_schema(statistics=train_stats)
tfdv.display_schema(schema)

# Análise de dados e validação com TFDV da base de teste
test_stats = tfdv.generate_statistics_from_dataframe(dataframe = test_set)

#checando anomalias nos dados de teste
anomalies = tfdv.validate_statistics(statistics = test_stats, schema = schema)

#Mostrando as anomalias detectadas
#Inteiros maiores do que 10
#Esperava o tipo STRING mas a coluna estava com o tipo INT
#Esperava o tipo FLOAT mas a coluna estava com o tipo INT
#Inteiros menores do que 0
print(tfdv.display_anomalies(anomalies)) #neste cenario nenhuma anomialia foi detectada

#criando novos dados com anomalias
test_set_copy=test_set.copy()
test_set_copy.drop("soot", axis=1, inplace=True)#Excluindo a coluna soot
print(test_set_copy.describe())

test_set_copy_stats=tfdv.generate_statistics_from_dataframe(dataframe = test_set_copy)

anomalies2 = tfdv.validate_statistics(statistics = test_set_copy_stats, schema = schema)

print(tfdv.display_anomalies(anomalies2))#'soot' Column dropped  Column is completely missing

#Preparação do esquema para produção (Serving)
schema.default_environment.append("TRAINING")
schema.default_environment.append("SERVING")

#Removendo a coluna alvo do esquema para produção
tfdv.get_feature(schema, "soot").not_in_environment.append("SERVING")

#Checando anomalias entre o ambiente em produção (Serving) e a nova base de teste
serving_env_anomalies = tfdv.validate_statistics(test_set_copy_stats, schema, environment = "SERVING")
print(tfdv.display_anomalies(serving_env_anomalies))

#salvando o esquema
tfdv.write_schema_text(schema = schema, output_path = "My_trained_models/pollution_schema.pbtxt")