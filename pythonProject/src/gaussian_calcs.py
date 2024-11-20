class GaussianCalcs:
    def __init__(self):
        pass

    """
    Formula de fatorial, a gente passa um numero e para cada proximo numero 
      a gente vai multiplicando
    """
    def factorial(self,n):
        if n == 0:
            return 1
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

    """
    Calculamos a exponencial, fazendo um loop com a quantidade de termos da serie de Taylor:
       como exponencial é uma aproximacao infinita o loop fica mais otimizado quando utilizamos o valor 10
       quando maior o numero de termos, mais otimizado sera o calculo da exponencial
    """
    def exp_manual(self,x, terms=10):
        result = 1.0
        term = 1.0
        for n in range(1, terms):
            term *= x / n
            result += term
        return result