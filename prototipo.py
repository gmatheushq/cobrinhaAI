import glob
import os
import random as rm

import joblib
import pandas as pd
import pygame as pyg

from features import (
    COLUNAS_MODELO,
    absoluto_de,
    direcao_atual,
    para_relativo,
)

# ==================================================
# CONFIGURACOES
# ==================================================

LARGURA = 800
ALTURA = 600
TAM = 20

COLUNAS = LARGURA // TAM
LINHAS = ALTURA // TAM

INTERVALO_MOVIMENTO = 120
TEMPO_REINICIO_IA = 800

# Partidas humanas com pontuacao menor que isso NAO sao salvas
# (imitar jogadas ruins ensina jogadas ruins). Use 0 para salvar tudo.
MIN_PONTUACAO_SALVAR = 0

PASTA_PROJETO = os.path.dirname(os.path.abspath(__file__))
CAMINHO_MODELO = os.path.join(PASTA_PROJETO, "modelo_cobrinha.pkl")

PASTA_DADOS = os.path.join(PASTA_PROJETO, "dadosTreino")
PASTA_DADOS_IA = os.path.join(PASTA_DADOS, "dadosIA")

# Os CSVs continuam sendo gravados em formato absoluto (dx, dy, direcao,
# perigos). O modelo usa colunas relativas a direcao da cobra, definidas
# em features.py (COLUNAS_MODELO), e preve Frente / Esquerda / Direita.

pyg.init()
janela = pyg.display.set_mode((LARGURA, ALTURA))
pyg.display.set_caption("Cobrinha")
relogio = pyg.time.Clock()

# ==================================================
# CORES E FONTES
# ==================================================

VERMELHO = (255, 0, 0)
VERDE = (0, 150, 0)
PRETO = (0, 0, 0)
AMARELO = (255, 220, 0)
BRANCO = (255, 255, 255)
AZUL = (0, 100, 255)
CINZA = (100, 100, 100)
CINZA_ESCURO = (40, 40, 40)

fonte_titulo = pyg.font.SysFont("Arial", 46, bold=True)
fonte_normal = pyg.font.SysFont("Arial", 28)
fonte_pequena = pyg.font.SysFont("Arial", 21)

# ==================================================
# BOTOES
# ==================================================

botao_humano = pyg.Rect(200, 255, 400, 70)
botao_ia = pyg.Rect(200, 345, 400, 70)

botao_ia_continua = pyg.Rect(100, 290, 290, 85)
botao_ia_parar = pyg.Rect(410, 290, 290, 85)
botao_voltar = pyg.Rect(300, 430, 200, 55)

botao_reiniciar = pyg.Rect(150, 390, 220, 60)
botao_menu = pyg.Rect(430, 390, 220, 60)

# ==================================================
# DIRECOES E TECLAS
# ==================================================

direcoes = {
    "Direita": (TAM, 0),
    "Esquerda": (-TAM, 0),
    "Cima": (0, -TAM),
    "Baixo": (0, TAM)
}

decisoes = {
    (TAM, 0): "Direita",
    (-TAM, 0): "Esquerda",
    (0, -TAM): "Cima",
    (0, TAM): "Baixo"
}

teclas = {
    pyg.K_RIGHT: (TAM, 0),
    pyg.K_LEFT: (-TAM, 0),
    pyg.K_UP: (0, -TAM),
    pyg.K_DOWN: (0, TAM),
    pyg.K_d: (TAM, 0),
    pyg.K_a: (-TAM, 0),
    pyg.K_w: (0, -TAM),
    pyg.K_s: (0, TAM)
}

# ==================================================
# CARREGAR MODELO
# ==================================================

def ha_dados_de_treino():
    """True se existe pelo menos um CSV na pasta dadosTreino (sem dadosIA)."""

    return bool(glob.glob(os.path.join(PASTA_DADOS, "*.csv")))


def resetar_modelo_se_sem_dados():
    """
    Se os dados de treino foram apagados, o modelo perde a base e e
    apagado tambem: a IA fica indisponivel ate um novo treino.
    Devolve True se o modelo foi resetado.
    """

    if ha_dados_de_treino():
        return False

    if os.path.exists(CAMINHO_MODELO):

        try:
            os.remove(CAMINHO_MODELO)
            print("Dados de treino apagados: modelo da IA resetado.")

        except OSError as erro:
            print("Nao foi possivel apagar o modelo:", erro)

    return True


modelo = None

if resetar_modelo_se_sem_dados():
    print("Sem dados de treino: a IA nao pode jogar. Grave partidas e rode o Treino.py.")

elif os.path.exists(CAMINHO_MODELO):
    try:
        modelo = joblib.load(CAMINHO_MODELO)

        # Garante que o modelo foi treinado com as mesmas colunas do jogo.
        colunas_modelo = list(getattr(modelo, "feature_names_in_", []))

        if colunas_modelo != COLUNAS_MODELO:
            print(
                "Modelo incompativel: foi treinado com outras colunas.\n"
                "Treine novamente com o Treino.py atualizado.\n"
                "Colunas do modelo:", colunas_modelo
            )
            modelo = None
        else:
            print("Modelo de IA carregado!")

    except Exception as erro:
        print("Nao foi possivel carregar o modelo:", erro)

else:
    print("Modelo nao encontrado. O modo humano continua disponivel.")


# ==================================================
# ESTADO DO JOGO
# ==================================================

tela_atual = "MENU"
modo_jogo = None

cobra = []
direcao = (TAM, 0)          # direcao escolhida para o proximo passo
direcao_passo = (TAM, 0)    # direcao do ultimo passo executado
fruta = None

momento = 0
pontuacao = 0
game_over = False
venceu = False

registros = []
dados_salvos = False

tempo_acumulado = 0
tempo_ate_reinicio = 0
numero_rodadas = 0


def eh_ia():
    return modo_jogo in ("IA_MANUAL", "IA_AUTO")


# ==================================================
# FEATURES E DECISAO DA IA
# ==================================================

def eh_perigo(celula, cobra_atual):
    """True se a celula e parede ou corpo da cobra."""

    fora = (
        celula[0] < 0
        or celula[0] >= LARGURA
        or celula[1] < 0
        or celula[1] >= ALTURA
    )

    if fora:
        return True

    # A ponta do rabo sai do lugar no proximo passo, entao nao conta.
    corpo = cobra_atual[:-1] if len(cobra_atual) > 1 else []

    return celula in corpo


def montar_features(cobra_atual, fruta_atual, direcao_atual):
    """Monta o dicionario de features (mesmas colunas de COLUNAS_MODELO)."""

    cabeca_x, cabeca_y = cobra_atual[0]

    if fruta_atual is not None:
        dx = (fruta_atual[0] - cabeca_x) // TAM
        dy = (fruta_atual[1] - cabeca_y) // TAM
    else:
        dx = 0
        dy = 0

    def perigo_em(nome):
        passo = direcoes[nome]
        celula = (cabeca_x + passo[0], cabeca_y + passo[1])
        return 1 if eh_perigo(celula, cobra_atual) else 0

    return {
        "dx": dx,
        "dy": dy,
        "dir_cima": 1 if direcao_atual == direcoes["Cima"] else 0,
        "dir_baixo": 1 if direcao_atual == direcoes["Baixo"] else 0,
        "dir_esquerda": 1 if direcao_atual == direcoes["Esquerda"] else 0,
        "dir_direita": 1 if direcao_atual == direcoes["Direita"] else 0,
        "perigo_cima": perigo_em("Cima"),
        "perigo_baixo": perigo_em("Baixo"),
        "perigo_esquerda": perigo_em("Esquerda"),
        "perigo_direita": perigo_em("Direita"),
        "TamanhoCobrinha": len(cobra_atual),
    }


def decidir_movimento(features):
    """
    Devolve a direcao que o modelo escolheu, sem nenhum filtro de
    seguranca: se ela mandar a cobra na parede ou no corpo, a cobra morre.
    (A regra de nao dar meia-volta continua sendo aplicada em
    atualizar_jogo, pois faz parte das regras do jogo.)
    """

    if modelo is None:
        return None

    # O modelo preve uma acao RELATIVA (Frente/Esquerda/Direita);
    # aqui ela vira uma direcao absoluta a partir da direcao atual.
    estado = pd.DataFrame([para_relativo(features)], columns=COLUNAS_MODELO)

    acao = modelo.predict(estado)[0]

    return absoluto_de(acao, direcao_atual(features))


# ==================================================
# GERAR FRUTA
# ==================================================

def gerar_fruta(cobra_atual):

    livres = [
        (x * TAM, y * TAM)
        for x in range(COLUNAS)
        for y in range(LINHAS)
        if (x * TAM, y * TAM) not in cobra_atual
    ]

    return rm.choice(livres) if livres else None


# ==================================================
# INICIAR PARTIDA
# ==================================================

def iniciar_jogo(modo=None):

    global tela_atual, modo_jogo, modelo
    global cobra, direcao, direcao_passo, fruta
    global momento, pontuacao, game_over, venceu
    global registros, dados_salvos, tempo_acumulado
    global tempo_ate_reinicio, numero_rodadas

    if modo is not None:
        modo_jogo = modo

    # Se os dados de treino foram apagados com o jogo aberto, reseta a IA.
    if eh_ia() and resetar_modelo_se_sem_dados():
        modelo = None

    if eh_ia() and modelo is None:

        print("Nao e possivel iniciar: modelo ausente, incompativel ou sem dados de treino.")

        tela_atual = "MENU"
        return

    numero_rodadas += 1
    tela_atual = "JOGO"

    cobra = [(400, 300)]
    direcao = (TAM, 0)
    direcao_passo = (TAM, 0)
    fruta = gerar_fruta(cobra)

    momento = 0
    pontuacao = 0
    game_over = False
    venceu = False

    registros = []
    dados_salvos = False

    tempo_acumulado = 0
    tempo_ate_reinicio = 0

    print(f"Rodada {numero_rodadas} iniciada. Modo: {modo_jogo}")


# ==================================================
# SALVAR DADOS HUMANOS E DADOS DA IA
# ==================================================

def salvar_dados():

    global dados_salvos

    # Impede que a mesma partida seja salva mais de uma vez.
    if dados_salvos:
        return

    dados_salvos = True

    # Escolher a pasta e o prefixo de acordo com o modo de jogo.
    if modo_jogo == "HUMANO":

        pasta_destino = PASTA_DADOS
        prefixo = "dados_humano"
        nome_tipo = "humanos"

        if pontuacao < MIN_PONTUACAO_SALVAR:

            print(
                f"Partida descartada (pontuacao {pontuacao} "
                f"< minimo {MIN_PONTUACAO_SALVAR})."
            )
            return

    elif modo_jogo == "IA_AUTO":

        pasta_destino = PASTA_DADOS_IA
        prefixo = "dados_ia"
        nome_tipo = "da IA"

    else:

        # O modo IA_MANUAL continua sem salvar dados.
        return

    if not registros:

        print("Nenhuma decisao foi registrada nesta partida.")
        return

    os.makedirs(pasta_destino, exist_ok=True)

    dados = pd.DataFrame(registros)

    # Encontrar o primeiro numero de arquivo ainda nao utilizado.
    numero = 1

    while os.path.exists(
        os.path.join(pasta_destino, f"{prefixo}_{numero}.csv")
    ):
        numero += 1

    caminho = os.path.join(
        pasta_destino,
        f"{prefixo}_{numero}.csv"
    )

    dados.to_csv(caminho, index=False)

    print(f"Dados {nome_tipo} salvos em:", caminho)
    print("Total de decisoes:", len(dados))


# ==================================================
# VALIDAR MOVIMENTO
# ==================================================

def movimento_valido(nova_direcao):

    # Compara com o ultimo passo EXECUTADO, para impedir inversao
    # mesmo se duas teclas forem apertadas entre dois passos.
    if len(cobra) > 1:

        inverso = (
            nova_direcao[0] == -direcao_passo[0]
            and nova_direcao[1] == -direcao_passo[1]
        )

        if inverso:
            return False

    return True


# ==================================================
# ATUALIZAR PARTIDA
# ==================================================

def atualizar_jogo():

    global direcao, direcao_passo, fruta, momento, pontuacao
    global game_over, venceu, tela_atual

    momento += 1

    # Features calculadas ANTES de aplicar a decisao deste passo.
    features = montar_features(cobra, fruta, direcao_passo)

    # A IA escolhe uma direcao usando o modelo treinado.
    if modo_jogo in ("IA_MANUAL", "IA_AUTO"):

        movimento = decidir_movimento(features)

        nova_direcao = direcoes.get(movimento, direcao)

        if movimento_valido(nova_direcao):
            direcao = nova_direcao

    cabeca_x, cabeca_y = cobra[0]

    nova_cabeca = (
        cabeca_x + direcao[0],
        cabeca_y + direcao[1]
    )

    comeu = nova_cabeca == fruta

    corpo_verificado = cobra if comeu else cobra[:-1]

    bateu_parede = (
        nova_cabeca[0] < 0
        or nova_cabeca[0] >= LARGURA
        or nova_cabeca[1] < 0
        or nova_cabeca[1] >= ALTURA
    )

    bateu_corpo = nova_cabeca in corpo_verificado

    # Registrar decisoes do humano e da IA no modo automatico.
    if modo_jogo in ("HUMANO", "IA_AUTO"):

        registro = dict(features)
        registro["decisao"] = decisoes[direcao]
        registro["Momento"] = momento  # so para referencia, nao e feature

        registros.append(registro)

    if bateu_parede or bateu_corpo:

        game_over = True

    else:

        cobra.insert(0, nova_cabeca)
        direcao_passo = direcao

        if comeu:

            pontuacao += 1
            fruta = gerar_fruta(cobra)

            if fruta is None:

                venceu = True
                game_over = True

        else:

            cobra.pop()

    if game_over:

        salvar_dados()

        if modo_jogo == "IA_AUTO":

            # A tela de intervalo mostra o resultado antes da proxima rodada.
            tela_atual = "REINICIO_IA"

        else:

            # Humano e IA em modo de teste ficam na tela final.
            tela_atual = "FIM"


# ==================================================
# FUNCOES DE DESENHO
# ==================================================

def desenhar_texto(texto, fonte, cor, x, y):

    imagem = fonte.render(texto, True, cor)
    janela.blit(imagem, (x, y))


def desenhar_texto_centralizado(texto, fonte, cor, x, y):

    imagem = fonte.render(texto, True, cor)
    janela.blit(imagem, imagem.get_rect(center=(x, y)))


def desenhar_botao(retangulo, texto, cor):

    pyg.draw.rect(janela, cor, retangulo, border_radius=12)

    desenhar_texto_centralizado(
        texto,
        fonte_pequena,
        BRANCO,
        retangulo.centerx,
        retangulo.centery
    )


def desenhar_menu():

    janela.fill(CINZA_ESCURO)

    desenhar_texto_centralizado(
        "JOGO DA COBRINHA",
        fonte_titulo,
        BRANCO,
        LARGURA // 2,
        135
    )

    desenhar_texto_centralizado(
        "Quem vai jogar?",
        fonte_normal,
        BRANCO,
        LARGURA // 2,
        205
    )

    desenhar_botao(
        botao_humano,
        "JOGAR COMO HUMANO",
        VERDE
    )

    desenhar_botao(
        botao_ia,
        "JOGAR COM A IA",
        AZUL if modelo is not None else CINZA
    )

    if modelo is None:

        desenhar_texto_centralizado(
            "Modelo de IA nao encontrado ou incompativel",
            fonte_pequena,
            AMARELO,
            LARGURA // 2,
            475
        )

    else:

        desenhar_texto_centralizado(
            "No modo humano, use as setas ou WASD",
            fonte_pequena,
            BRANCO,
            LARGURA // 2,
            475
        )


def desenhar_escolha_ia():

    janela.fill(CINZA_ESCURO)

    desenhar_texto_centralizado(
        "COMO A IA DEVE JOGAR?",
        fonte_titulo,
        BRANCO,
        LARGURA // 2,
        150
    )

    desenhar_texto_centralizado(
        "Escolha o comportamento quando ela perder:",
        fonte_pequena,
        BRANCO,
        LARGURA // 2,
        220
    )

    desenhar_botao(
        botao_ia_continua,
        "JOGAR SOZINHA",
        VERDE
    )

    desenhar_botao(
        botao_ia_parar,
        "PARAR AO PERDER",
        AZUL
    )

    desenhar_botao(
        botao_voltar,
        "VOLTAR",
        CINZA
    )

    desenhar_texto_centralizado(
        "Jogar sozinha: reinicia automaticamente apos cada derrota.",
        fonte_pequena,
        BRANCO,
        LARGURA // 2,
        510
    )


def desenhar_jogo():

    janela.fill(VERDE)

    if fruta is not None:

        pyg.draw.rect(
            janela,
            AMARELO,
            (fruta[0], fruta[1], TAM, TAM)
        )

    for i, segmento in enumerate(cobra):

        cor = VERMELHO if i == 0 else (180, 0, 0)

        pyg.draw.rect(
            janela,
            cor,
            (segmento[0], segmento[1], TAM, TAM)
        )

    nome_jogador = {

        "HUMANO": "Humano",
        "IA_MANUAL": "IA - modo de teste",
        "IA_AUTO": "IA - modo continuo"

    }.get(modo_jogo, "")

    desenhar_texto(
        f"Pontuacao: {pontuacao}",
        fonte_normal,
        BRANCO,
        10,
        10
    )

    desenhar_texto(
        f"Jogador: {nome_jogador}",
        fonte_pequena,
        BRANCO,
        10,
        48
    )

    if modo_jogo == "IA_AUTO":

        desenhar_texto(
            f"Rodada: {numero_rodadas} | ESC para parar",
            fonte_pequena,
            BRANCO,
            10,
            78
        )


def desenhar_fim():

    janela.fill(CINZA_ESCURO)

    mensagem = "VOCE VENCEU!" if venceu else "GAME OVER"

    cor_titulo = AMARELO if venceu else BRANCO

    desenhar_texto_centralizado(
        mensagem,
        fonte_titulo,
        cor_titulo,
        LARGURA // 2,
        170
    )

    desenhar_texto_centralizado(
        f"Pontuacao final: {pontuacao}",
        fonte_normal,
        BRANCO,
        LARGURA // 2,
        245
    )

    if modo_jogo == "HUMANO":

        if pontuacao >= MIN_PONTUACAO_SALVAR:
            explicacao = "Suas decisoes foram salvas em dadosTreino."
        else:
            explicacao = "Partida abaixo do minimo: nao foi salva."

    else:

        explicacao = "A IA parou para voce analisar esta rodada."

    desenhar_texto_centralizado(
        explicacao,
        fonte_pequena,
        BRANCO,
        LARGURA // 2,
        310
    )

    desenhar_botao(
        botao_reiniciar,
        "JOGAR NOVAMENTE",
        VERDE
    )

    desenhar_botao(
        botao_menu,
        "MENU INICIAL",
        AZUL
    )


def desenhar_intervalo_ia():

    janela.fill(CINZA_ESCURO)

    mensagem = "IA VENCEU!" if venceu else "IA PERDEU!"

    desenhar_texto_centralizado(
        mensagem,
        fonte_titulo,
        AMARELO,
        LARGURA // 2,
        170
    )

    desenhar_texto_centralizado(
        f"Pontuacao desta rodada: {pontuacao}",
        fonte_normal,
        BRANCO,
        LARGURA // 2,
        245
    )

    desenhar_texto_centralizado(
        f"Rodada {numero_rodadas} concluida",
        fonte_pequena,
        BRANCO,
        LARGURA // 2,
        305
    )

    desenhar_texto_centralizado(
        "Iniciando a proxima rodada automaticamente...",
        fonte_pequena,
        BRANCO,
        LARGURA // 2,
        365
    )

    desenhar_texto_centralizado(
        "Pressione ESC para interromper e voltar ao menu",
        fonte_pequena,
        AMARELO,
        LARGURA // 2,
        430
    )


# ==================================================
# LOOP PRINCIPAL
# ==================================================

rodando = True

while rodando:

    tempo = relogio.tick(60)

    for evento in pyg.event.get():

        if evento.type == pyg.QUIT:

            if tela_atual == "JOGO":
                salvar_dados()

            rodando = False
            continue

        # ESC interrompe a execucao continua da IA.
        if (
            evento.type == pyg.KEYDOWN
            and evento.key == pyg.K_ESCAPE
            and modo_jogo == "IA_AUTO"
        ):

            # Salvar a partida atual antes de interromper a IA.
            salvar_dados()

            tela_atual = "MENU"
            modo_jogo = None

            continue

        if tela_atual == "MENU":

            if evento.type == pyg.MOUSEBUTTONDOWN and evento.button == 1:

                if botao_humano.collidepoint(evento.pos):

                    numero_rodadas = 0
                    iniciar_jogo("HUMANO")

                elif botao_ia.collidepoint(evento.pos) and modelo is not None:

                    tela_atual = "ESCOLHA_IA"

        elif tela_atual == "ESCOLHA_IA":

            if evento.type == pyg.MOUSEBUTTONDOWN and evento.button == 1:

                if botao_ia_continua.collidepoint(evento.pos):

                    numero_rodadas = 0
                    iniciar_jogo("IA_AUTO")

                elif botao_ia_parar.collidepoint(evento.pos):

                    numero_rodadas = 0
                    iniciar_jogo("IA_MANUAL")

                elif botao_voltar.collidepoint(evento.pos):

                    tela_atual = "MENU"

        elif tela_atual == "JOGO":

            if (
                modo_jogo == "HUMANO"
                and evento.type == pyg.KEYDOWN
                and evento.key in teclas
            ):

                nova_direcao = teclas[evento.key]

                if movimento_valido(nova_direcao):
                    direcao = nova_direcao

        elif tela_atual == "FIM":

            if evento.type == pyg.MOUSEBUTTONDOWN and evento.button == 1:

                if botao_reiniciar.collidepoint(evento.pos):

                    iniciar_jogo()

                elif botao_menu.collidepoint(evento.pos):

                    tela_atual = "MENU"
                    modo_jogo = None

    # Atualizacao normal da cobrinha.
    if tela_atual == "JOGO":

        tempo_acumulado += tempo

        if tempo_acumulado >= INTERVALO_MOVIMENTO:

            tempo_acumulado = 0
            atualizar_jogo()

    # Pequena pausa entre rodadas no modo automatico.
    elif tela_atual == "REINICIO_IA":

        tempo_ate_reinicio += tempo

        if tempo_ate_reinicio >= TEMPO_REINICIO_IA:

            iniciar_jogo()

    # Desenhar tela atual.
    if tela_atual == "MENU":

        desenhar_menu()

    elif tela_atual == "ESCOLHA_IA":

        desenhar_escolha_ia()

    elif tela_atual == "JOGO":

        desenhar_jogo()

    elif tela_atual == "FIM":

        desenhar_fim()

    elif tela_atual == "REINICIO_IA":

        desenhar_intervalo_ia()

    pyg.display.flip()


# Se o jogo for fechado durante qualquer partida em andamento,
# salva os dados.
if tela_atual == "JOGO":

    salvar_dados()

pyg.quit()

print("Jogo encerrado!")