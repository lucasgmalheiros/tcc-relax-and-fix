from functions_model_sy import main_solve

# Parâmetros
    # Time limit
TIME_LIMIT = 1800

# Arquivos
    # Arquivo de salvamento
results_csv = f'resultados_n_plantas_produto.csv'
    # Pasta com instâncias a serem resolvidas
folder_instancias = f'../instancias/fix_produtos/'

# Chama o solver
main_solve(folder_instancias, results_csv, TIME_LIMIT)
