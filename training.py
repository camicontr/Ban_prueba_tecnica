from spark_pipeline import *
from pycaret.classification import setup, create_model, compare_models, tune_model, plot_model, finalize_model, predict_model

import pandas as pd
import sklearn
sklearn.set_config(enable_metadata_routing=True)


def run_pipeline():
    spark = initialize_spark()
    
    file_names = {
        'prueba_op_base_pivot_var_rpta_alt_enmascarado_trtest.csv': 'df_var_rpta',
        'prueba_op_master_customer_data_enmascarado_completa.csv': 'df_master_customer_data'
    }
    
    dataframes = load_data(spark, file_names)
    df_var_rpta = preprocess_var_rpta(dataframes['df_var_rpta'])
    df_master_customer_data = preprocess_master_customer_data(dataframes['df_master_customer_data'])
    df_final = merge_dataframes(df_var_rpta, df_master_customer_data)
    
    output_path = os.path.join('data_processed', 'processed_data.csv') 
    # df_final.write.csv(output_path, header=True, mode='overwrite') #guardar con spark
    df_final_pandas = df_final.toPandas()
    df_final_pandas.to_csv(output_path, index=False)
        

def training_model():
    df = pd.read_csv('data_processed/processed_data.csv')
    df_model = df[['var_rpta_alt', 'min_mora', 'vlr_obligacion', 'endeudamiento', 'promesas_cumplidas', 'rpc', 'pago_mes', 'banca_index', 'segmento_index', 'producto_index', 'alternativa_aplicada_agr_index', 'marca_alternativa_index', 'cant_acuerdo_binario_index', 'cant_alter_posibles_index', 'tipo_vivienda_index', 'tot_patrimonio', 'egresos_mes', 'total_ing', 'tot_activos', 'tot_pasivos', 'personas_dependientes']]
    model_setup = setup(
        data = df_model,
        target = 'var_rpta_alt',
        train_size = 0.8,
        numeric_features = ['min_mora', 'vlr_obligacion', 'endeudamiento', 'promesas_cumplidas', 'rpc', 'pago_mes', 'tot_patrimonio', 'egresos_mes', 'total_ing', 'tot_activos', 'tot_pasivos', 'personas_dependientes'],
        categorical_features = ['banca_index', 'segmento_index', 'producto_index', 'alternativa_aplicada_agr_index', 'marca_alternativa_index', 'cant_acuerdo_binario_index', 'cant_alter_posibles_index', 'tipo_vivienda_index'],
        remove_outliers = True,
        pca = True,
        pca_components = 0.9,
        remove_multicollinearity = True,
        multicollinearity_threshold = 0.9,
        fold = 10
    )
    print(model_setup)
    
    

if __name__ == "__main__":
    #run_pipeline()
    training_model()
