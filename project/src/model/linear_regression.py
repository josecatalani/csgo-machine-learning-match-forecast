import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

pd.set_option('future.no_silent_downcasting', True)

class LinearRegressionModel:
    def predict(self):
        ## Definindo as colunas que serão usadas para treinar o modelo
        team_map_results_columns_to_add  = ["Id", "Kills", "Deaths", "PlusMinus", "Adr", "Kast", "Rating"]
        team_map_results_columns_to_drop = ["Id", "Kills", "Deaths", "Adr", "Kast"]

        map_results = [f"map{map_num}Team{team_num}Side{side}Player{player_num}{attribute}"
                        for map_num in range(1, 6) 
                        for team_num in range(1, 3) 
                        for player_num in range(1, 6)
                        for attribute in team_map_results_columns_to_add
                        for side in ["Both", "CounterTerrorist", "Terrorist"]]

        ## Definindo as colunas que serão retiradas para a análise
        drop_columns = [f"map{map_num}Team{team_num}Side{side}Player{player_num}{attribute}"
                        for map_num in range(1, 6) 
                        for team_num in range(1, 3) 
                        for player_num in range(1, 6)
                        for attribute in team_map_results_columns_to_drop
                        for side in ["Both", "CounterTerrorist", "Terrorist"]]

        ### Listando as colunas que serão usadas das partidas, como: Jogadores, Mortes, Assistências etc.
        matches_train_data_columns = ['eventId', 'matchId', 'mapBestOf'] + map_results

        matches_results_train_data = pd.read_csv('data/raw/matches_results.csv')
        matches_train_data = pd.read_csv('data/raw/matches.csv')

        matches_train_data = matches_train_data[matches_train_data_columns]

        matches_results_train_data_columns = ['eventId', 'matchId', 'TeamOneScore', 'TeamTwoScore', 'teamOneWon', 'teamTwoWon']
        matches_results_train_data = matches_results_train_data[[col for col in matches_results_train_data.columns if any(s in col for s in matches_results_train_data_columns)]]

        ## Combinando os dados gerais das partidas com os detalhes das partidas
        full_matches_train_data = pd.merge(matches_results_train_data, matches_train_data, on='matchId', how='inner')

        ## Data Wrangling
        full_matches_train_data.fillna(0, inplace=True)
        full_matches_train_data.replace("Not Available", 0, inplace=True)
        full_matches_train_data.drop(['eventId_y', 'eventId_x', 'matchId'], axis=1, inplace=True)

        full_matches_train_data.drop(drop_columns, axis=1, inplace=True)
        full_matches_train_data.drop(['mapBestOf'], axis=1, inplace=True)

        # Features (colunas que serão utilizadas para fazer a previsão)
        X = full_matches_train_data.drop(['teamOneWon', 'teamTwoWon'], axis=1)
        X = X.apply(pd.to_numeric, errors='coerce')

        # Target (coluna que será prevista)
        y = full_matches_train_data['teamOneWon']
        y = y.apply(pd.to_numeric, errors='coerce')

        print(f"Número de amostras em X?{X.shape} e y:{y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

        # Treinar o modelo com a base de treino
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)

        # Fazer previsões com o conjunto de teste
        y_pred = model.predict(X_test)

        # Calcular a acurácia
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Acurácia do Modelo: {accuracy:.2f}')
        print(f"Score do modelo: {model.score(X_test, y_test)}")