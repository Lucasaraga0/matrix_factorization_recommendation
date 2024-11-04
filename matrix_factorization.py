import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class MatrixFactorization():
    def __init__(self, number_caracteristics=50, learning_rate=0.1, max_iter=1000, regularization=0.01, momentum=0.9):        
        self.number_caracteristics = number_caracteristics # tamanho de cada embedding 
        self.learning_rate = learning_rate # learning rate do gradiente descendente 
        self.max_iter = max_iter 
        self.regularization = regularization #fator de regularização 
        self.momentum = momentum  # inicialmente eu não tinha colocado o termo de momentum 

    def criar_matrizes(self, Matriz_R):

        # cria as matrizes de usuário e item que vao ser usadas para a criacao da matriz r_hat
        self.user_matrix = np.random.random(size=(Matriz_R.shape[0], self.number_caracteristics))
        self.item_matrix = np.random.random(size=(Matriz_R.shape[1], self.number_caracteristics))
        
    def criar_matriz_R_hat(self):

        self.r_hat = np.dot(self.user_matrix, self.item_matrix.T)
    
    def calcular_erro(self, Matriz_R):

        erro = (Matriz_R - (self.user_matrix @ self.item_matrix.T)) 
        mse = np.mean(erro ** 2)
        return np.sqrt(mse)


    def calcular_gradientes(self, Matriz_R):

        # aqui é a ideia de criar uma máscara para os valores que não foram avaliados pelos usuários
        # retirado do github https://github.com/wangyuhsin/matrix-factorization/blob/main/README.md   
        
        array_r = np.array(Matriz_R)
        mask = np.where(array_r != 0, 1, 0)
        n = mask.sum() 

        pred_user = (self.user_matrix @ self.item_matrix.T) * mask
        pred_item = (self.item_matrix @ self.user_matrix.T).T * mask

        grad_user = -2 /n * np.dot ((array_r - pred_user), self.item_matrix)
        grad_itens = -2/n * np.dot((array_r - pred_item).T, self.user_matrix)
        return grad_user, grad_itens
    

    def treinar(self, Matriz_R, mostrar_erro=False):
        array_R = np.array(Matriz_R)        

        self.erros_iteracoes = []
 
        v_user = np.zeros_like(self.user_matrix)
        v_items = np.zeros_like(self.item_matrix)

        for iter in range(self.max_iter):
            
            user_grad, items_grad = self.calcular_gradientes(Matriz_R)

            v_user = self.momentum * v_user + self.regularization * user_grad 
            v_items = self.momentum * v_items + self.regularization * items_grad
            
            self.user_matrix -= self.learning_rate * v_user
            self.item_matrix -= self.learning_rate * v_items

            if mostrar_erro:
                    if iter %10 == 0:
                        erro_atual = self.calcular_erro(array_R)
                        print(f"Iteração: {iter + 1}, Erro: {erro_atual}") 
                        self.erros_iteracoes.append(erro_atual)
      
        
        self.r_hat = np.dot(self.user_matrix, self.item_matrix.T)
        return self.r_hat

    def fit(self, Matriz_R, mostrar_erro=False, salvar = False):
        self.criar_matrizes(Matriz_R)
        self.criar_matriz_R_hat()  
        if salvar: 
            np.save("matriz_r_hat",self.r_hat)
            np.save("matriz_m1", self.user_matrix)
            np.save("matriz_m2", self.item_matrix)
            R_hat_df = pd.DataFrame(self.r_hat, columns= Matriz_R.columns)
            R_hat_df.to_csv("Matriz_R_hat.csv")
        return self.treinar(Matriz_R, mostrar_erro)



def pegar_recomendacoes(user_number, number_recomendations, matriz_r, matrizr_hat = None, prints = True):
    if matrizr_hat is None:
        Mf = MatrixFactorization(number_caracteristics= number_recomendations*5)
        Mf.fit(matriz_r)
        matrizr_hat = pd.DataFrame(Mf.r_hat, columns= matriz_r.columns)
    
    #tendo a matriz de recomendacoes r_hat basta pegar os que tem maior avaliacao e não foram avaliadas pelo usuario
    
    real = matriz_r[user_number: user_number+1]
    rec = matrizr_hat[user_number: user_number+1]
    
    for column in real.columns:
        if real.loc[user_number, column] != 0:
            rec.loc[user_number, column] = 0
    lista_rec = rec.iloc[0].nlargest(number_recomendations).index.tolist()
    if prints:
        print("Buscando recomendações para você ...")
        print("Talvez você goste de:")
        
        for i,element in enumerate(lista_rec):
            if (i+1) %3 != 0 and i+1 != len(lista_rec) :
                print(element, end = ", ")
            else:
                print(element)
    return lista_rec
