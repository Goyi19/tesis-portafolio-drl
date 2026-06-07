import pandas as pd
import os

class DataHandler:
    """
    Clase encargada de la gestión, partición y entrega de datos para el 
    entorno de portafolios DRL.
    """
    
    def __init__(self, file_path='data/processed/source_data.csv'):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No se encontró el dataset en {file_path}. Verifica la carpeta data/processed/")
        
        self.df = pd.read_csv(file_path)
        
        # Definición de columnas de observación (Features para la IA)
        self.feature_cols = [
            'norm_mu_mix_spx', 
            'norm_mu_mix_cmc', 
            'norm_var_mix_spx', 
            'norm_var_mix_cmc', 
            'norm_cov_mix'
        ]
        
        # Definición de columnas de mercado (Para cálculo de Reward y Capital)
        self.market_cols = [
            'ret_spx', 
            'ret_cmc',
            'mu_mix_spx', 
            'mu_mix_cmc', 
            'var_mix_spx', 
            'var_mix_cmc', 
            'cov_mix'
        ]

    def get_partition(self, mode='train'):
        """
        Retorna el slice del dataframe correspondiente a la partición solicitada.
        Train: 1-114 (iloc 0:114)
        Val: 115-138 (iloc 114:138)
        Test: 139-163 (iloc 138:163)
        """
        if mode == 'train':
            return self.df.iloc[0:114].copy()
        elif mode == 'val':
            return self.df.iloc[114:138].copy()
        elif mode == 'test':
            return self.df.iloc[138:163].copy()
        else:
            raise ValueError("Modo no reconocido. Usa 'train', 'val' o 'test'.")

    def get_observation_space_dim(self):
        """Retorna la dimensión de las características de mercado."""
        return len(self.feature_cols)

if __name__ == "__main__":
    # Script de prueba rápida (Sanity Check)
    handler = DataHandler()
    train_df = handler.get_partition('train')
    val_df = handler.get_partition('val')
    test_df = handler.get_partition('test')
    
    print(f"Carga exitosa:")
    print(f"- Filas Train: {len(train_df)} (esperado 114)")
    print(f"- Filas Val:   {len(val_df)} (esperado 24)")
    print(f"- Filas Test:  {len(test_df)} (esperado 25)")
    print(f"- Total:       {len(train_df)+len(val_df)+len(test_df)} (esperado 163)")
    print(f"\nFeatures detectadas: {handler.feature_cols}")
