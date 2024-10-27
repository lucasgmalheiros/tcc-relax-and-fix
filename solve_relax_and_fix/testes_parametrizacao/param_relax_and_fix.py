from solve_relax_and_fix.functions_model_sy_rf import model_sy_relaxation
import csv
import numpy as np
from gurobipy import GRB

# Arquivo de salvamento
results_csv = 'param_rf.csv'
# Pasta com instâncias a serem resolvidas
folder_ = '../instancias/maria_desiree/'
# Instâncias
instancia_ = 'AAA0121570_4.dat'

window = [1, 2, 3, 4, 6]
overlap = [0, 1, 2, 3, 4]

# for file in instancias:
# Teste de parametrização
for y in overlap:
    for w in window:
        if w > y:
            # Criação do modelo
            model = model_sy_relaxation(str(folder_), str(instancia_))
            m, I, J, T, Z = model[0], model[1], model[2], model[3], model[4]
            
            m.setParam('OutputFlag', 0)  # Desativa a exibição de logs
            K = np.ceil((len(T) - w) / (w - y)) + 1  # Número total de iterações
            m.Params.timelimit = 1800 / K  # Tempo limite por subproblema (s)
            runtime = 0
            status_list = []

            # Auxiliares da heurística
            k, to, tf = 1, 0, w
            N = T[-1]  # Último período do horizonte de planejamento

            # Relaxar domínio das variáveis binárias (Z_ijt)
            for key in Z.keys():
                Z[key].setAttr("VType", GRB.CONTINUOUS)
                Z[key].setAttr("LB", 0)
                Z[key].setAttr("UB", 1)
            m.update()

            print('=' * 20)
            print(f'w = {w} | y = {y}')
            print('k:', k, 'to:', to, 'tf:', tf)

            # Main loop
            while tf <= N:
                # Variáveis na janela tornam-se binárias
                for i in I:
                    for j in J:
                        for t in range(to, tf):
                            Z[i, j, t].setAttr("VType", GRB.BINARY)
                m.update()

                # Resolução do subprolema
                m.optimize()
                runtime += m.Runtime
                status_list.append(m.Status)

                # Fixação das soluções obtidas considerando overlap
                for i in I:
                    for j in J:
                        for t in range(to, tf - y):
                            Z[i, j, t].setAttr("LB", Z[i, j, t].X)  # Lower bound fixado para a solução encontrada
                            Z[i, j, t].setAttr("UB", Z[i, j, t].X)  # Upper bound fixado para a solução encontrada
                m.update()

                # Atualização dos auxiliares
                k += 1
                to = tf - y
                tf = min(tf + w - y, N + 1)

                print('k:', k, 'to:', to, 'tf:', tf)

            # Última iteração
            # Variáveis na janela tornam-se binárias
            for i in I:
                for j in J:
                    for t in range(to, tf):
                        Z[i, j, t].setAttr("VType", GRB.BINARY)
            m.update()

            # Última resolução
            m.optimize()
            runtime += m.Runtime
            status_list.append(m.Status)
            
            # Parâmetros armazenáveis da resolução
            inst = m.ModelName
            lb = round(m.ObjBound, 2)
            ub = round(m.ObjVal, 2)
            gap = round(m.MIPgap, 2)
            time = round(runtime, 2)
            if 3 in status_list:
                status = 3
            else:
                status = max(status_list)

            # Verificação de status de resolução se resultado deve ser guardado:
            if status not in [11]:  # Não escreve caso seja interrompido pelo usuário 
                # Escrever ou adicionar resultados no CSV
                with open(results_csv, mode='a', newline='') as file:
                    writer = csv.writer(file)

                    # Escrever o cabeçalho apenas na primeira vez
                    if file.tell() == 0:  # Se o arquivo estiver vazio, escrever o cabeçalho
                        writer.writerow(['Instancia', 'Windows', 'Overlap', 'Lower Bound (LB)',
                                            'Upper Bound (UB)', 'Gap', 'Tempo (s)', 'Status'])

                    # Adicionar os resultados da instância ao CSV
                    writer.writerow([inst, w, y, lb, ub, gap, time, status])
