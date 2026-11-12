"""
Gera o arvore.html: a visualizacao estilizada da floresta, num unico arquivo.

Uso:
    python exportar_arvore.py

Le o modelo_cobrinha.pkl (gerado pelo Treino.py) e o arvore_template.html,
e escreve o arvore.html ao lado deles. O arquivo final nao depende de nada
alem do navegador, entao basta publicar o link.

Nao altera Treino.py, Teste.py nem prototipo.py. Depois de treinar de novo,
rode este script outra vez para atualizar a pagina.
"""

import json
import os

import joblib
import numpy as np

from features import COLUNAS_MODELO

PASTA_PROJETO = os.path.dirname(os.path.abspath(__file__))
CAMINHO_MODELO = os.path.join(PASTA_PROJETO, "modelo_cobrinha.pkl")
CAMINHO_TEMPLATE = os.path.join(PASTA_PROJETO, "arvore_template.html")
CAMINHO_SAIDA = os.path.join(PASTA_PROJETO, "arvore.html")

MARCADOR = "/*__DADOS__*/null"

# Probabilidades guardadas como inteiros de 0 a 10000 (4 casas), para o HTML ficar leve.
ESCALA_PROB = 10000


def exportar_arvore(estimador):
    """Uma arvore em colunas: o filho esquerdo de i e sempre i+1, entao so guardamos o direito."""

    t = estimador.tree_
    n_nos = t.node_count

    for i in range(n_nos):
        if t.children_left[i] != -1 and t.children_left[i] != i + 1:
            raise ValueError("Ordem de nos inesperada: o filho esquerdo nao e o proximo no.")

    valores = t.value[:, 0, :].astype(float)
    soma = valores.sum(axis=1, keepdims=True)
    soma[soma == 0] = 1
    probs = valores / soma

    return {
        "f": [int(x) for x in t.feature],
        "t": [round(float(x), 4) if t.feature[i] >= 0 else 0 for i, x in enumerate(t.threshold)],
        "r": [int(x) for x in t.children_right],
        "n": [int(x) for x in t.n_node_samples],
        "w": [int(round(float(x))) for x in t.weighted_n_node_samples],
        "p": [int(round(float(x) * ESCALA_PROB)) for x in probs.flatten()],
    }


def main():
    modelo = joblib.load(CAMINHO_MODELO)

    colunas = list(COLUNAS_MODELO)

    if hasattr(modelo, "feature_names_in_") and list(modelo.feature_names_in_) != colunas:
        raise ValueError(
            "As colunas do modelo nao batem com features.COLUNAS_MODELO: "
            f"{list(modelo.feature_names_in_)} != {colunas}"
        )

    dados = {
        "classes": [str(c) for c in modelo.classes_],
        "colunas": colunas,
        "escala": ESCALA_PROB,
        "arvores": [exportar_arvore(e) for e in modelo.estimators_],
    }

    with open(CAMINHO_TEMPLATE, encoding="utf-8") as f:
        template = f.read()

    if MARCADOR not in template:
        raise ValueError(f"O template nao tem o marcador {MARCADOR}")

    json_dados = json.dumps(dados, separators=(",", ":"))

    with open(CAMINHO_SAIDA, "w", encoding="utf-8") as f:
        f.write(template.replace(MARCADOR, json_dados))

    total_nos = sum(len(a["f"]) for a in dados["arvores"])
    tamanho_mb = os.path.getsize(CAMINHO_SAIDA) / 1e6

    print(f"{len(dados['arvores'])} arvores, {total_nos} nos")
    print(f"Pagina gerada: {CAMINHO_SAIDA} ({tamanho_mb:.2f} MB)")


if __name__ == "__main__":
    main()
