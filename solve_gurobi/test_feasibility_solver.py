from functions_model_sy import lista_instancias, model_sy
import csv

def main_solve_feasibility(folder_path, results_file, time_limit):
    for file in lista_instancias(folder_path, results_file):
        model = model_sy(folder_path, file)
        m, I, J, T = model[0], model[1], model[2], model[3]

        # Resolução
        # Critérios de parada
        m.Params.timelimit = time_limit  # Tempo (s) (default = inf)
        m.Params.SolutionLimit = 1
        # m.Params.MIPgap = 0.01  # Gap de otimalidade (default = 0.0001 = 0.01%)
        # Otimização
        m.optimize()

        # Parâmetros armazenáveis da resolução
        inst = m.ModelName
        plantas = len(J)
        produtos = len(I)
        periodos = len(T)
        lb = round(m.ObjBound, 4)
        try:
            ub = round(m.ObjVal, 4)
        except AttributeError:
            ub = 'inf'
        gap = round(m.MIPgap, 4)
        time = round(m.Runtime, 4)
        status = m.Status

        # Verificação de status de resolução se resultado deve ser guardado:
        if status not in [11]:  # Não escreve caso seja interrompido pelo usuário 
            # Escrever ou adicionar resultados no CSV
            with open(results_file, mode='a', newline='') as file:
                writer = csv.writer(file)

                # Escrever o cabeçalho apenas na primeira vez
                if file.tell() == 0:  # Se o arquivo estiver vazio, escrever o cabeçalho
                    writer.writerow(['Instancia', 'Plantas', 'Produtos', 'Periodos', 
                                    'Lower Bound (LB)', 'Upper Bound (UB)', 'Gap', 
                                    'Tempo (s)', 'Status', 'N_solutions'])

                # Adicionar os resultados da instância ao CSV
                writer.writerow([inst, plantas, produtos, periodos, lb, ub, gap, time, status, m.SolCount])


# Parâmetros
TIME_LIMIT = 1800

# Arquivos
    # Arquivo de salvamento
results_csv = f'classify_factibilidade.csv'
    # Pasta com instâncias a serem resolvidas
folder_instancias = f'../instancias/testes_factibilidade/'

# Chama o solver
main_solve_feasibility(folder_instancias, results_csv, TIME_LIMIT)
