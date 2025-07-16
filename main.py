# Passo 1: Importar Bibliotecas Necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from graphviz import Source

# Passo 2: Carregar e Preparar o Dataset
file_path = "Test.csv"  # substitua pelo nome do seu arquivo
data = pd.read_csv(file_path)
print("Dados originais:")
print(data.head())

# Passo 2.2 - Converter valores categóricos para numéricos
label_encoder = LabelEncoder()
for column in data.columns:
    data[column] = label_encoder.fit_transform(data[column])

print("\nDados após a preparação:")
print(data.head())

# Passo 2.3 - Definir X e y
X = data.drop(columns=["Segmentation"])  # target é a coluna "Segmentation"
y = data["Segmentation"]

# Passo 3: Dividir dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Passo 4: Treinar o Modelo
clf = DecisionTreeClassifier(criterion="gini", random_state=42)
clf.fit(X_train, y_train)

# Passo 5: Avaliar o Modelo
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAcurácia: {accuracy:.2f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# Passo 6: Exportar e Visualizar a Árvore
# Aqui é importante colocar os nomes corretos das classes
dot_data = export_graphviz(
    clf,
    feature_names=X.columns,
    class_names=["A", "B", "C", "D"],  # adaptado ao seu dataset
    filled=True,
    rounded=True,
    special_characters=True
)
graph = Source(dot_data)
graph.render("decision_tree_customer_segmentation", format="png")
graph.view()
