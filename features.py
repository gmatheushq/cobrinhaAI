"""
Funcoes compartilhadas entre prototipo.py e Treino.py.

O prototipo.py grava os CSVs em formato ABSOLUTO (dx, dy, direcao e perigos
em Cima/Baixo/Esquerda/Direita). O modelo, porem, aprende em formato
RELATIVO a direcao da cobra: "a fruta esta a frente / a esquerda" e
"tem perigo a frente / a esquerda / a direita". A acao prevista tambem e
relativa: Frente, Esquerda ou Direita.

Assim a licao "fruta a esquerda -> vire a esquerda" vale para qualquer
direcao em que a cobra esteja indo.
"""

# Colunas gravadas nos CSVs pelo prototipo.py (formato absoluto).
COLUNAS_ABSOLUTAS = [
    "dx",
    "dy",
    "dir_cima",
    "dir_baixo",
    "dir_esquerda",
    "dir_direita",
    "perigo_cima",
    "perigo_baixo",
    "perigo_esquerda",
    "perigo_direita",
    "TamanhoCobrinha",
]

# Colunas que o modelo usa (formato relativo). Mesma ordem no treino e no jogo.
COLUNAS_MODELO = [
    "fruta_frente",      # > 0: fruta adiante | < 0: fruta atras
    "fruta_esq",         # > 0: fruta a esquerda | < 0: fruta a direita
    "rel_perigo_frente",
    "rel_perigo_esq",
    "rel_perigo_dir",
    "TamanhoCobrinha",
]

ACOES = ("Frente", "Esquerda", "Direita")

# Vetores em celulas (tela: y cresce para baixo).
VETORES = {
    "Direita": (1, 0),
    "Esquerda": (-1, 0),
    "Cima": (0, -1),
    "Baixo": (0, 1),
}

COLUNA_DIRECAO = {
    "Cima": "dir_cima",
    "Baixo": "dir_baixo",
    "Esquerda": "dir_esquerda",
    "Direita": "dir_direita",
}

COLUNA_PERIGO = {
    "Cima": "perigo_cima",
    "Baixo": "perigo_baixo",
    "Esquerda": "perigo_esquerda",
    "Direita": "perigo_direita",
}


def _esquerda(v):
    # Direita(1,0) -> Cima(0,-1) -> Esquerda(-1,0) -> Baixo(0,1)
    return (v[1], -v[0])


def _direita(v):
    return (-v[1], v[0])


def _nome_do_vetor(v):
    for nome, vetor in VETORES.items():
        if vetor == v:
            return nome
    return None


def direcao_atual(f):
    """Nome da direcao em que a cobra ia (lida das colunas dir_*)."""

    for nome, coluna in COLUNA_DIRECAO.items():
        if f[coluna] == 1:
            return nome

    return "Direita"


def para_relativo(f):
    """Converte as features absolutas de um passo para o formato do modelo."""

    nome = direcao_atual(f)

    frente = VETORES[nome]
    esq = _esquerda(frente)
    dire = _direita(frente)

    dx = f["dx"]
    dy = f["dy"]

    return {
        "fruta_frente": dx * frente[0] + dy * frente[1],
        "fruta_esq": dx * esq[0] + dy * esq[1],
        "rel_perigo_frente": f[COLUNA_PERIGO[nome]],
        "rel_perigo_esq": f[COLUNA_PERIGO[_nome_do_vetor(esq)]],
        "rel_perigo_dir": f[COLUNA_PERIGO[_nome_do_vetor(dire)]],
        "TamanhoCobrinha": f["TamanhoCobrinha"],
    }


def acao_relativa(nome_direcao, decisao):
    """
    Converte a decisao absoluta (ex.: "Cima") em Frente/Esquerda/Direita
    em relacao a direcao em que a cobra ia. Devolve None para meia-volta.
    """

    frente = VETORES[nome_direcao]
    escolhido = VETORES.get(decisao)

    if escolhido is None:
        return None

    if escolhido == frente:
        return "Frente"

    if escolhido == _esquerda(frente):
        return "Esquerda"

    if escolhido == _direita(frente):
        return "Direita"

    return None


def absoluto_de(acao, nome_direcao):
    """Converte a acao relativa do modelo em direcao absoluta (nome)."""

    frente = VETORES[nome_direcao]

    if acao == "Esquerda":
        return _nome_do_vetor(_esquerda(frente))

    if acao == "Direita":
        return _nome_do_vetor(_direita(frente))

    return nome_direcao
