import glob
import os

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from features import (
    COLUNAS_ABSOLUTAS,
    COLUNAS_MODELO,
    acao_relativa,
    direcao_atual,
    para_relativo,
)

# ==================================================
# CONFIGURACOES
# ==================================================

PASTA_PROJETO = os.path.dirname(os.path.abspath(__file__))
PASTA_DADOS = os.path.join(PASTA_PROJETO, "dadosTreino")
CAMINHO_MODELO = os.path.join(PASTA_PROJETO, "modelo_cobrinha.pkl")

COLUNA_ALVO = "decisao"
SEMENTE = 54

# ==================================================
# LER OS CSVs (so a pasta principal; dadosIA nao entra)
# ==================================================

arquivos_csv = sorted(glob.glob(os.path.join(PASTA_DADOS, "*.csv")))

if not arquivos_csv:
    raise FileNotFoundError("Nenhum arquivo CSV encontrado na pasta dadosTreino!")

lista_dados = []

for arquivo in arquivos_csv:
    nome = os.path.basename(arquivo)
    print("Lendo:", nome)

    parte = pd.read_csv(arquivo)

    faltando = [
        c for c in COLUNAS_ABSOLUTAS + [COLUNA_ALVO] if c not in parte.columns
    ]

    if faltando:
        raise ValueError(
            f"O arquivo {nome} nao tem as colunas {faltando}. "
            "Parece um CSV antigo: mova-o para fora de dadosTreino."
        )

    # Cada arquivo e uma partida: usado so para dividir treino/teste.
    parte["partida"] = nome

    lista_dados.append(parte)

dados = pd.concat(lista_dados, ignore_index=True)
dados = dados.dropna(subset=COLUNAS_ABSOLUTAS + [COLUNA_ALVO])

print("\nTotal de partidas:", dados["partida"].nunique())
print("Total de registros:", len(dados))

# ==================================================
# CONVERTER PARA O FORMATO RELATIVO (frente / esquerda / direita)
# ==================================================

linhas = []
alvos = []
grupos = []

for registro, partida in zip(
    dados[COLUNAS_ABSOLUTAS + [COLUNA_ALVO]].to_dict("records"),
    dados["partida"],
):

    acao = acao_relativa(direcao_atual(registro), registro[COLUNA_ALVO])

    # Meia-volta (so possivel com a cobra de tamanho 1): ignorada.
    if acao is None:
        continue

    linhas.append(para_relativo(registro))
    alvos.append(acao)
    grupos.append(partida)

X = pd.DataFrame(linhas, columns=COLUNAS_MODELO)
y = pd.Series(alvos, name=COLUNA_ALVO)
grupos = pd.Series(grupos, name="partida")

print("\nDistribuicao das acoes (relativas):")
print(y.value_counts())
print((y.value_counts(normalize=True) * 100).round(1).astype(str) + " %")

if (y == "Frente").mean() > 0.9:
    print(
        "\nAviso: mais de 90% das jogadas sao 'Frente'. "
        "O modelo pode aprender pouco sobre quando virar."
    )


def criar_modelo():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        random_state=SEMENTE,
        class_weight="balanced",
        n_jobs=-1,
    )


# ==================================================
# VALIDACAO: PARTIDAS INTEIRAS NO TREINO OU NO TESTE
# ==================================================

if grupos.nunique() >= 5:

    divisor = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEMENTE)
    idx_treino, idx_teste = next(divisor.split(X, y, groups=grupos))

    X_train, X_test = X.iloc[idx_treino], X.iloc[idx_teste]
    y_train, y_test = y.iloc[idx_treino], y.iloc[idx_teste]

    print(
        f"\nPartidas no treino: {grupos.iloc[idx_treino].nunique()} | "
        f"no teste: {grupos.iloc[idx_teste].nunique()}"
    )

    modelo_validacao = criar_modelo()

    print("\nTreinando IA (validacao)...")
    modelo_validacao.fit(X_train, y_train)

    acc_treino = modelo_validacao.score(X_train, y_train)
    acc_teste = modelo_validacao.score(X_test, y_test)

    print(f"\nAcuracia treino: {acc_treino:.4f}")
    print(f"Acuracia teste:  {acc_teste:.4f}")

    if acc_treino - acc_teste > 0.15:
        print("Aviso: grande diferenca treino/teste -> possivel overfitting.")

    classes = list(modelo_validacao.classes_)

    if set(y_test.unique()) == set(classes):

        probabilidades = modelo_validacao.predict_proba(X_test)

        auc = roc_auc_score(
            y_test,
            probabilidades,
            multi_class="ovr",
            average="weighted",
            labels=classes,
        )

        print(f"AUC: {auc:.4f}")
        print(f"Gini: {2 * auc - 1:.4f}")

    else:
        print("AUC ignorado: nem todas as acoes aparecem no teste.")

    print("\nRESULTADO NO TESTE:")
    print(classification_report(y_test, modelo_validacao.predict(X_test)))

else:
    print(
        "\nMenos de 5 partidas: validacao ignorada "
        "(grave mais partidas para avaliar o modelo)."
    )

# ==================================================
# MODELO FINAL: TREINA COM TODAS AS PARTIDAS E SALVA
# ==================================================

modelo = criar_modelo()

print("\nTreinando modelo final com todos os dados...")
modelo.fit(X, y)

joblib.dump(modelo, CAMINHO_MODELO, compress=3)

tamanho_mb = os.path.getsize(CAMINHO_MODELO) / 1e6

print("\nModelo treinado e salvo com sucesso!")
print("Arquivo:", CAMINHO_MODELO)
print(f"Tamanho: {tamanho_mb:.2f} MB")

# ==================================================
# ATUALIZA A PAGINA arvore.html COM O MODELO RECEM-TREINADO
# (o modelo ja foi salvo; se a exportacao falhar, nada se perde)
# ==================================================

try:
    import exportar_arvore

    exportar_arvore.main()

except Exception as erro:
    print("\nNao consegui atualizar o arvore.html:", erro)

# ==================================================
# VISUALIZACAO DA ARVORE DE DECISAO
# (o modelo ja foi salvo; se a visualizacao falhar, nada se perde)
# ==================================================


try:
    from visualizar_arvore import mostrar_arvore

    mostrar_arvore(modelo, COLUNAS_MODELO)

except ImportError as erro:
    print(
        "\nVisualizacao ignorada (falta o matplotlib ou o "
        "visualizar_arvore.py):", erro
    )
