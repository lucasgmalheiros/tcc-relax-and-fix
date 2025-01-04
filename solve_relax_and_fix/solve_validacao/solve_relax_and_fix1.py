import sys
sys.path.append(r'../')
from functions_model_sy_rf import main_solve_rf

# Parâmetros
    # Time limit
TIME_LIMIT = 1800

# Arquivos
    # Arquivo de salvamento
results_csv = f'resultados1.csv'
    # Pasta com instâncias a serem resolvidas
folder_instancias = f'../../instancias/validar_modelo/'

params = [(1, 0), (6, 5), (2, 0), (6, 4)]

for param in params:
    w, y = param
    main_solve_rf(folder_instancias, results_csv, TIME_LIMIT, window_size=w, overlap_size=y)
