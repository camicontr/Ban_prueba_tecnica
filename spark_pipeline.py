from pyspark.sql.functions import col, sum, count, to_date, concat_ws, lit, row_number, when
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.stat import Correlation
from pyspark.sql.window import Window

import os


def index_categorical_columns(df, categorical_cols):
    indexers = [StringIndexer(inputCol=col, outputCol=col+"_index").fit(df) for col in categorical_cols]
    for indexer in indexers:
        df = indexer.transform(df)
    return df


def cast_columns_to_double(df, columns):
    for column in columns:
        df = df.withColumn(column, col(column).cast("double"))
    return df


def get_latest_records(df, partition_col, order_col1, order_col2):
    window_spec = Window.partitionBy(partition_col).orderBy(col(order_col1).desc(), col(order_col2).desc())
    
    df = df.withColumn("row_number", row_number().over(window_spec))
    
    df_latest = df.filter(col("row_number") == 1).drop("row_number")
    
    return df_latest



spark = SparkSession.builder.appName("Data_Analysis").getOrCreate()

file_names = {
    'prueba_op_base_pivot_var_rpta_alt_enmascarado_trtest.csv': 'df_var_rpta',
    'prueba_op_master_customer_data_enmascarado_completa.csv': 'df_master_customer_data'
}

dataframes = {}
for file_name, df_name in file_names.items():
    file_path = os.path.join('data', file_name)
    dataframes[df_name] = spark.read.csv(file_path, header=True, inferSchema=True)

df_var_rpta = dataframes['df_var_rpta']
df_master_customer_data = dataframes['df_master_customer_data']

df_var_rpta = df_var_rpta.dropDuplicates(["nit_enmascarado", "num_oblig_enmascarado", "fecha_var_rpta_alt", "cant_alter_posibles"])

select_num_col = ['promesas_cumplidas', 'rpc', 'endeudamiento', 'min_mora', 'vlr_obligacion']
df_var_rpta = cast_columns_to_double(df_var_rpta, select_num_col)
df_var_rpta = df_var_rpta.dropna(subset=select_num_col)

select_categorical_cols = ["tipo_var_rpta_alt", "banca", "segmento", "producto", "alternativa_aplicada_agr", "marca_alternativa", "cant_acuerdo_binario", "cant_alter_posibles"]

df_var_rpta = df_var_rpta.withColumn(
    "alternativa_aplicada_agr",
    when(col("alternativa_aplicada_agr") == "None", "Sin Alternativa")
    .otherwise(col("alternativa_aplicada_agr"))
)

df_var_rpta = df_var_rpta.withColumn(
    "marca_alternativa",
    when(col("marca_alternativa") == "N.A", "Sin Información")
    .otherwise(col("marca_alternativa"))
)

df_var_rpta = index_categorical_columns(df_var_rpta, select_categorical_cols)

categorical_cols_indexed = [col+"_index" for col in select_categorical_cols]
final_cols = ["nit_enmascarado", "num_oblig_orig_enmascarado", "num_oblig_enmascarado", "var_rpta_alt"] + select_num_col + categorical_cols_indexed

df_var_rpta = df_var_rpta.select(final_cols)

df_master_customer_data.groupBy("nit_enmascarado").count().filter("count > 1").show()

df_master_customer_data = get_latest_records(df_master_customer_data, "nit_enmascarado", "year", "month")

customer_cols = ["nit_enmascarado", "tot_patrimonio", "egresos_mes", "tipo_vivienda", "personas_dependientes", "total_ing", "tot_activos", "tot_pasivos"]

df_master_customer_data = df_master_customer_data.select(customer_cols)

df_master_customer_data = df_master_customer_data.withColumn(
    "tipo_vivienda",
    when((col("tipo_vivienda") == "None") | (col("tipo_vivienda") == "NO INFORMA"), "Desconocido")
    .otherwise(col("tipo_vivienda"))
)

df_master_customer_data = index_categorical_columns(df_master_customer_data, ['tipo_vivienda'])
df_master_customer_data = cast_columns_to_double(df_master_customer_data, ["tot_patrimonio", "egresos_mes", "personas_dependientes", "total_ing", "tot_activos", "tot_pasivos"])

df_final = df_var_rpta.join(
    df_master_customer_data,
    on=["nit_enmascarado"],
    how="left"
)

df_final = df_final.filter(col("tot_activos").isNotNull())
