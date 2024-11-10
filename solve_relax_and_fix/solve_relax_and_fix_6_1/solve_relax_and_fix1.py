import sys
sys.path.append(r'../')
from functions_model_sy_rf import main_solve_rf

# Parâmetros
    # Conjunto de instâncias (1 a 4)
CONJUNTO = 1
    # Time limit
TIME_LIMIT = 1800

# Arquivos
    # Arquivo de salvamento
results_csv = f'resultados{CONJUNTO}.csv'
    # Pasta com instâncias a serem resolvidas
folder_instancias = f'../../instancias/conjunto{CONJUNTO}/'

w = 6  # Window
y = 1  # Overlap

main_solve_rf(folder_instancias, results_csv, TIME_LIMIT, window_size=w, overlap_size=y)
