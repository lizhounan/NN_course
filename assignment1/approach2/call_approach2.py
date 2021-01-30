from approach2_fixed import train_process
import numpy as np
import pandas as pd
matrix = []
dev = [0, 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
for d in dev:
    samples, fhs, fhas = train_process('Gray', 10, 100, False, 10, d, 0)
    # c = input('continue? [y/ n]')
    # if c == 'y' or c == 'Y':
    #     continue
    # else:
    #     break
    
    matrix.append(fhs)
    matrix.append(fhas)
matrix = np.transpose(np.array(matrix))
samples = np.array(samples)

pd.DataFrame(matrix, dtype='float').to_excel("matrix_approach2_fixed_training_set.xlsx")
pd.DataFrame(samples).to_excel("samples.xlsx")

print(matrix)
print(samples)