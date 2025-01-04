import sys
sys.path.append(r'../')
from functions_model_sy_rf import main_solve_rf

# Parâmetros
    # Time limit
TIME_LIMIT = 1800

# Arquivos
    # Arquivo de salvamento
results_csv = f'resultados4.csv'
    # Pasta com instâncias a serem resolvidas
folder_instancias = f'../../instancias/validar_modelo/'

params = [(4, 0), (4, 3), (4, 1), (4, 2)]

for param in params:
    w, y = param
    main_solve_rf(folder_instancias, results_csv, TIME_LIMIT, window_size=w, overlap_size=y)
