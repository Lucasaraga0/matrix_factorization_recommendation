# matrix_factorization_recommendation

PT-BR: Sistema de recomendação baseados no algoritmo de Matrix Factorization. Abaixo tem-se um exemplo de utilização da implementação.

## Exemplo

```python
import numpy as np
import pandas as pd
import matrix_factorization as MF

Matriz_R = pd.read_csv("movies_pivot_1.csv")
Matriz_R_hat = pd.read_csv("Matriz_R_hat.csv")

User_number = np.random.randint(low = 0, high= Matriz_R.shape[0])
MF.pegar_recomendacoes(user_number= User_number, number_recomendations= 10, matriz_r= Matriz_R)
```
```
Buscando recomendações para você ...
Talvez você goste de:
Toy Story (1995), GoldenEye (1995), Four Rooms (1995)
Get Shorty (1995), Copycat (1995), Shanghai Triad (Yao a yao yao dao waipo qiao) (1995)
Twelve Monkeys (1995), Babe (1995), Dead Man Walking (1995)
Richard III (1995)
```
