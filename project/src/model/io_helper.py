import pandas as pd

class IOHelper:
    def __init__(self, filepath):
        """
        Inicializa a classe com o caminho do arquivo.
        :param filepath: caminho para o arquivo CSV.
        """
        self.filepath = filepath

    def write(self, df):
        """
        Escreve o DataFrame fornecido em um arquivo CSV.
        :param df: DataFrame a ser escrito no arquivo.
        """
        try:
            df.to_csv(self.filepath, index=False)
            print(f"DataFrame foi escrito com sucesso em {self.filepath}")
        except Exception as e:
            print(f"Erro ao escrever o DataFrame: {e}")

    def read(self):
        """
        Lê um arquivo CSV e retorna um DataFrame.
        :return: DataFrame lido do arquivo CSV.
        """
        try:
            df = pd.read_csv(self.filepath)
            print(f"DataFrame foi lido com sucesso de {self.filepath}")
            return df
        except Exception as e:
            print(f"Erro ao ler o arquivo CSV: {e}")
            return None
