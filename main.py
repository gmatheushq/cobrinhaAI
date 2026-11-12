import pygame as pyg
import time as tm
import random as rm
import pandas as pd


DadosParaTreino = pd.DataFrame(
    columns=['Momento', 'decisão', 'estaVivo', 'Ganhou', 'DistanObjetivo']
)  


x, y = 400, 300
numerosX = list(range(0, 780, 20))
numerosY = list(range(0, 580, 20))

pyg.init()

vermelho = (255, 0, 0)
cor1 = (0, 255, 100)  

janela = pyg.display.set_mode((800, 600))
pyg.display.set_caption("Cobrinha para IA")  

Xfruta = rm.choice(numerosX)  
Yfruta = rm.choice(numerosY)

tamanhoCobrinha = 1
coordenadaCobrinhaFinal = (x, y)
coordenadaCobrinhaInicial = (x, y)
MOMENTOdeJogo = 0
decision = 0
isLife = 1
WinGame = 0
DistanObjetivo = 0

rodando = True

while rodando:

    janela.fill((0, 200, 0))

    # Atualiza as coordenadas da cobrinha
    coordenadaCobrinhaInicial = (x, y)  

    DistanObjetivo = (
        abs(Xfruta - coordenadaCobrinhaInicial[0])
        + abs(Yfruta - coordenadaCobrinhaInicial[1])
    )  

    # Registra o estado atual do jogo
    DadosParaTreino.loc[len(DadosParaTreino)] = [
        MOMENTOdeJogo, decision, isLife, WinGame, DistanObjetivo
    ]  

    # Verifica se a cobrinha comeu a fruta
    if coordenadaCobrinhaInicial == (Xfruta, Yfruta):  
        tamanhoCobrinha += 1

        Xfruta = rm.choice(numerosX)  
        Yfruta = rm.choice(numerosY)

    # Processa os eventos
    for evento in pyg.event.get():

        if evento.type == pyg.QUIT:
            rodando = False

        if evento.type == pyg.KEYDOWN:

            if evento.key == pyg.K_UP:
                y -= 20
                decision = "Cima"  #gpt: registra a decisão tomada

            elif evento.key == pyg.K_LEFT:
                x -= 20
                decision = "Esquerda"  #gpt: registra a decisão tomada

            elif evento.key == pyg.K_RIGHT:
                x += 20
                decision = "Direita"  #gpt: registra a decisão tomada

            elif evento.key == pyg.K_DOWN:
                y += 20
                decision = "Baixo"  #gpt: registra a decisão tomada

    # Verifica se a cobrinha saiu da tela
    if x > 780 or x < 0 or y > 580 or y < 0:  #gpt: verificação fora do processamento de eventos
        if isLife == 1:  #gpt: evita imprimir Game Over repetidamente
            print("Game Over")

        isLife = 0  #gpt: atualiza o estado de vida

    # Desenha a fruta
    pyg.draw.rect(
        janela, cor1, (Xfruta, Yfruta, 20, 20)
    )  #gpt: a fruta agora aparece na tela

    # Desenha a cobrinha
    pyg.draw.rect(janela, vermelho, (x, y, 20, 20))

    # Atualiza a janela
    pyg.display.flip()

    MOMENTOdeJogo += 1

    tm.sleep(0.02)  #gpt: limita a velocidade aproximada do laço

# Salva os dados coletados
DadosParaTreino.to_csv(
    "DadosParaTreino.csv", index=False
)  #gpt: exporta os dados para uso posterior na IA

pyg.quit()
