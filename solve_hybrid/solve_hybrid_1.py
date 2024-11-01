from functions_model_sy_hybrid import main_solve_hybrid_matheuristic

# Parâmetros
    # Conjunto de instâncias (1 a 4)
CONJUNTO = 1
    # Time limit
TIME_LIMIT = 1800

# Arquivos
    # Arquivo de salvamento
results_csv = f'resultados{CONJUNTO}.csv'
    # Pasta com instâncias a serem resolvidas
folder_instancias = f'../conjunto{CONJUNTO}/'

main_solve_hybrid_matheuristic(folder_instancias, results_csv, TIME_LIMIT, 20, 3)
