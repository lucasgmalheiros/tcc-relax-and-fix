import sys
import csv
import numpy as np
from gurobipy import GRB
sys.path.append(r'../')
from functions_model_sy_rf import model_sy_relaxation, lista_instancias

def main_solve_feasibility_rf(folder_path, results_file, time_limit, window_size, overlap_size):
    for file in lista_instancias(folder_path, results_file):
        # Criação do modelo
        print(file)
        model = model_sy_relaxation(folder_path, file)
        m, I, J, T, Z = model[0], model[1], model[2], model[3], model[4]

        m.setParam('OutputFlag', 0)  # Desativa a exibição de logs
        m.params.SoftMemLimit = 8  # Limite de 8GB de RAM
        m.Params.SolutionLimit = 1
        K = np.ceil((len(T) - window_size) / (window_size - overlap_size)) + 1  # Número total de iterações
        m.Params.timelimit = time_limit / K  # Tempo limite por subproblema (s)
        runtime = 0
        status_list = []

        # Auxiliares da heurística
        k, to, tf = 1, 0, window_size
        N = T[-1]  # Último período do horizonte de planejamento

        print('=' * 20)
        print(f'w = {window_size} | y = {overlap_size}')
        print('k:', k, 'to:', to, 'tf:', tf - 1)

        # Main loop
        while tf <= N:
            # Variáveis na janela tornam-se binárias
            for i in I:
                for j in J:
                    for t in range(to, tf):
                        Z[i, j, t].setAttr("VType", GRB.BINARY)

            # Resolução do subprolema
            m.optimize()
            print('N_sols: ', m.SolCount)
            try:
                print('UB:', m.ObjVal)
            except AttributeError:
                print('UB:', 'inf')
            runtime += m.Runtime
            status_list.append(m.Status)
            if m.Status == GRB.INFEASIBLE or m.SolCount == 0:
                break

            # Fixação das soluções obtidas considerando overlap
            for i in I:
                for j in J:
                    for t in range(to, tf - overlap_size):
                        Z[i, j, t].setAttr("LB", Z[i, j, t].X)
                        Z[i, j, t].setAttr("UB", Z[i, j, t].X)

            # Atualização dos auxiliares
            k += 1
            to = tf - overlap_size
            tf = min(tf + window_size - overlap_size, N + 1)

            print('k:', k, 'to:', to, 'tf:', tf - 1)

        # Última iteração
        if m.Status != GRB.INFEASIBLE and m.SolCount > 0:
            # Variáveis na janela tornam-se binárias
            for i in I:
                for j in J:
                    for t in range(to, tf):
                        Z[i, j, t].setAttr("VType", GRB.BINARY)
            # Na última iteração quero apenas contar soluções possíveis
            m.Params.SolutionLimit = 2_000_000_000
            m.Params.PoolSolutions = 1_000_000
            m.Params.PoolSearchMode = 2
            m.Params.timelimit = 600
            # Última resolução
            m.optimize()
            runtime += m.Runtime
            status_list.append(m.Status)

        # Parâmetros armazenáveis da resolução
        inst = m.ModelName
        lb = round(m.ObjBound, 4)
        try:
            ub = round(m.ObjVal, 4)
        except AttributeError:
            ub = 'inf'
        gap = round(m.MIPgap, 4)
        time = round(runtime, 4)
        if 3 in status_list:
            status = 3
        elif m.SolCount == 0:
            status = 99
        else:
            status = max(status_list)

        # Verificação de status de resolução se resultado deve ser guardado:
        if status not in [11]:  # Não escreve caso seja interrompido pelo usuário 
            # Escrever ou adicionar resultados no CSV
            with open(results_file, mode='a', newline='') as file:
                writer = csv.writer(file)

                # Escrever o cabeçalho apenas na primeira vez
                if file.tell() == 0:  # Se o arquivo estiver vazio, escrever o cabeçalho
                    writer.writerow(['Instancia', 'Plantas', 'Produtos', 'Windows', 'Overlap', 'Iterations', 'Lower Bound (LB)',
                                        'Upper Bound (UB)', 'Gap', 'Tempo (s)', 'Status_firsts', 'Status_last', 'N_solutions'])

                # Adicionar os resultados da instância ao CSV
                writer.writerow([inst, len(J), len(I), window_size, overlap_size, k, lb, ub, gap, time, status, m.Status, m.SolCount])


# Parâmetros
TIME_LIMIT = 100

# Arquivos
    # Arquivo de salvamento
results_csv = f'analise_factibilidade.csv'
    # Pasta com instâncias a serem resolvidas
folder_instancias = f'../../instancias/testes_factibilidade/'

w = 1  # Window
y = 0  # Overlap

main_solve_feasibility_rf(folder_instancias, results_csv, TIME_LIMIT, window_size=w, overlap_size=y)
