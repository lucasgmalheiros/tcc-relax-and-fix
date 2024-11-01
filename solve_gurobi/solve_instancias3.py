from functions_model_sy import main_solve

# Parâmetros
    # Conjunto de instâncias (1 a 4)
CONJUNTO = 3
    # Time limit
TIME_LIMIT = 1800

# Arquivos
    # Arquivo de salvamento
results_csv = f'resultados{CONJUNTO}.csv'
    # Pasta com instâncias a serem resolvidas
folder_instancias = f'../instancias/conjunto{CONJUNTO}/'

# Chama o solver
main_solve(folder_instancias, results_csv, TIME_LIMIT)
