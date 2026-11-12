"""
Visualizador interativo da floresta de decisao (chamado pelo Treino.py).

O modelo e uma Random Forest: muitas arvores de decisao votando. A janela
mostra UMA arvore por vez e deixa voce:

  * trocar de arvore (botoes ou setas <- ->);
  * mostrar mais ou menos niveis, descendo pelos nos (botoes ou setas cima/baixo);
  * arrastar com o mouse para mover e usar a roda do mouse para dar zoom
    (o texto das caixas cresce junto, entao da para "mergulhar" na arvore);
  * trocar entre o modo DIDATICO (perguntas em portugues) e o modo TECNICO
    (parecido com o plot_tree do scikit-learn: gini, samples, value);
  * montar um cenario (onde esta a fruta, onde ha perigo) e ver o CAMINHO
    que a arvore percorre em vermelho e como as arvores votam.

Sem ambiente grafico (ex.: ssh), salva duas imagens PNG em vez de abrir a janela.
"""

import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.widgets import Button, CheckButtons, Slider

PASTA_PROJETO = os.path.dirname(os.path.abspath(__file__))

COR_ACAO = {
    "Frente": "#4C78A8",
    "Esquerda": "#F58518",
    "Direita": "#54A24B",
}

TEXTO_ACAO = {
    "Frente": "Seguir em frente",
    "Esquerda": "Virar à esquerda",
    "Direita": "Virar à direita",
}

# Geometria das caixas (em unidades do desenho)
LARG = 0.92
ALT = 0.66
PASSO_Y = 1.3


def clarear(cor, quanto):
    """quanto = 0 -> cor pura; 1 -> branco."""

    r, g, b = to_rgb(cor)

    return (r + (1 - r) * quanto, g + (1 - g) * quanto, b + (1 - b) * quanto)


def pergunta(feature, limiar):
    """
    Traduz o teste de um no ("feature <= limiar") para uma pergunta em
    portugues. Devolve (texto, sim_e_ramo_esquerdo).
    O ramo esquerdo da arvore e o do teste VERDADEIRO (feature <= limiar).
    """

    n = int(abs(limiar))

    if feature == "fruta_frente":

        if limiar < 0:
            texto = "Fruta está atrás?" if n == 0 else f"Fruta está a mais de {n} casas atrás?"
            return texto, True

        texto = "Fruta está à frente?" if n == 0 else f"Fruta está a mais de {n} casas à frente?"
        return texto, False

    if feature == "fruta_esq":

        if limiar < 0:
            texto = "Fruta está à direita?" if n == 0 else f"Fruta está a mais de {n} casas à direita?"
            return texto, True

        texto = "Fruta está à esquerda?" if n == 0 else f"Fruta está a mais de {n} casas à esquerda?"
        return texto, False

    lados = {
        "rel_perigo_frente": "à frente",
        "rel_perigo_esq": "à esquerda",
        "rel_perigo_dir": "à direita",
    }

    if feature in lados:
        return f"Tem perigo {lados[feature]}?", False

    if feature == "TamanhoCobrinha":
        return f"A cobra tem mais de {int(limiar)} casas?", False

    return f"{feature} > {limiar:g}?", False


class Visualizador:

    def __init__(self, modelo, colunas):

        self.modelo = modelo
        self.colunas = list(colunas)
        self.classes = [str(c) for c in modelo.classes_]
        self.n_arvores = len(modelo.estimators_)

        self.indice = 0
        self.nivel = 3
        self.modo = "didatico"

        self.arrasto = None
        self.textos = []
        self.patches = {}
        self.estilo_padrao = {}
        self.arestas = {}

        # As setas do teclado sao usadas aqui; tira o conflito com a barra padrao.
        for nome in ("keymap.back", "keymap.forward"):
            atual = plt.rcParams.get(nome, [])
            plt.rcParams[nome] = [k for k in atual if k not in ("left", "right")]

        self._montar_figura()
        self._desenhar_arvore(recentralizar=True)
        self._atualizar_cenario()

    # ==================================================
    # FIGURA E CONTROLES
    # ==================================================

    def _montar_figura(self):

        fig = plt.figure(figsize=(17, 9.5))
        self.fig = fig

        try:
            fig.canvas.manager.set_window_title("Como a cobrinha decide")
        except Exception:
            pass

        fig.text(
            0.02, 0.975,
            f"Como a cobrinha decide: uma floresta de {self.n_arvores} árvores",
            fontsize=17, weight="bold", va="top",
        )

        self.txt_ajuda = fig.text(0.02, 0.935, "", fontsize=10.5, va="top", color="#333333")
        self.txt_info = fig.text(0.02, 0.882, "", fontsize=12, weight="bold", va="top", color="#222222")

        self.ax = fig.add_axes([0.02, 0.13, 0.70, 0.74])

        fig.legend(
            handles=[
                Patch(color=clarear(COR_ACAO.get(c, "#888888"), 0.25), label=TEXTO_ACAO.get(c, c))
                for c in self.classes
            ],
            loc="upper right",
            bbox_to_anchor=(0.72, 0.89),
            ncol=len(self.classes),
            frameon=False,
            fontsize=10.5,
        )

        fig.text(
            0.02, 0.108,
            "Mouse: arraste para mover · roda do mouse = zoom (o texto cresce)   |   "
            "Teclas: ← → trocam de árvore · ↑ ↓ mostram mais/menos níveis · M troca o modo",
            fontsize=9.5, color="#666666", va="center",
        )

        # ----- Botoes de baixo -----

        def botao(x, largura, rotulo, acao):
            eixo = fig.add_axes([x, 0.025, largura, 0.055])
            b = Button(eixo, rotulo)
            b.label.set_fontsize(10.5)
            b.on_clicked(acao)
            return b

        self.b_ant = botao(0.02, 0.14, "◀ Árvore anterior", lambda _e: self._trocar_arvore(-1))
        self.b_prox = botao(0.17, 0.14, "Árvore seguinte ▶", lambda _e: self._trocar_arvore(+1))
        self.b_menos = botao(0.33, 0.12, "− Menos níveis", lambda _e: self._mudar_nivel(-1))
        self.b_mais = botao(0.46, 0.12, "+ Mais níveis", lambda _e: self._mudar_nivel(+1))
        self.b_centro = botao(0.59, 0.10, "Centralizar", lambda _e: self._centralizar())
        self.b_modo = botao(0.74, 0.24, "", lambda _e: self._alternar_modo())

        # ----- Painel de cenario (direita) -----

        fig.text(0.76, 0.965, "Teste você mesmo", fontsize=14, weight="bold", va="top")
        fig.text(
            0.76, 0.935,
            "Mude o cenário: o caminho da árvore\naparece em vermelho e as árvores votam.",
            fontsize=9.5, va="top", color="#444444",
        )

        self.txt_s1 = fig.text(0.76, 0.872, "", fontsize=10.5)
        self.s1 = Slider(fig.add_axes([0.76, 0.842, 0.21, 0.02]), "", -20, 20, valinit=0, valstep=1)

        self.txt_s2 = fig.text(0.76, 0.815, "", fontsize=10.5)
        self.s2 = Slider(fig.add_axes([0.76, 0.785, 0.21, 0.02]), "", -20, 20, valinit=3, valstep=1)

        self.txt_s3 = fig.text(0.76, 0.758, "", fontsize=10.5)
        self.s3 = Slider(fig.add_axes([0.76, 0.728, 0.21, 0.02]), "", 1, 30, valinit=3, valstep=1)

        for s in (self.s1, self.s2, self.s3):
            s.valtext.set_visible(False)
            s.on_changed(self._ao_mudar_cenario)

        self.chk = CheckButtons(
            fig.add_axes([0.76, 0.58, 0.21, 0.12]),
            ["Perigo à frente", "Perigo à esquerda", "Perigo à direita"],
            [False, False, False],
        )

        for rotulo in self.chk.labels:
            rotulo.set_fontsize(11)

        self.chk.on_clicked(self._ao_mudar_cenario)

        self.ax_votos = fig.add_axes([0.78, 0.30, 0.19, 0.22])

        self.txt_dec = fig.text(0.865, 0.205, "", ha="center", va="top", fontsize=14, weight="bold")

        fig.text(
            0.865, 0.115,
            "A decisão real soma as probabilidades\nde todas as árvores (quase sempre\nigual à maioria dos votos).",
            ha="center", va="top", fontsize=8.5, color="#666666",
        )

        # ----- Eventos -----

        fig.canvas.mpl_connect("scroll_event", self._ao_rolar)
        fig.canvas.mpl_connect("button_press_event", self._ao_apertar)
        fig.canvas.mpl_connect("button_release_event", self._ao_soltar)
        fig.canvas.mpl_connect("motion_notify_event", self._ao_mover)
        fig.canvas.mpl_connect("key_press_event", self._ao_tecla)
        fig.canvas.mpl_connect("resize_event", self._ao_redimensionar)

        self._atualizar_botao_modo()
        self._atualizar_ajuda()
        self._atualizar_rotulos_cenario()

    # ==================================================
    # LAYOUT E DESENHO DA ARVORE
    # ==================================================

    def _arvore(self):
        return self.modelo.estimators_[self.indice]

    def _layout(self, t):

        pos = {}
        tipo = {}
        arestas = []
        proximo = [0]

        def rec(no, prof):

            esq = t.children_left[no]
            dire = t.children_right[no]

            if esq == -1:
                tipo[no] = "folha"
            elif prof >= self.nivel:
                tipo[no] = "cortado"
            else:
                tipo[no] = "interno"

            if tipo[no] == "interno":
                xe = rec(esq, prof + 1)
                xd = rec(dire, prof + 1)
                arestas.append((no, esq, True))
                arestas.append((no, dire, False))
                x = (xe + xd) / 2
            else:
                x = float(proximo[0])
                proximo[0] += 1

            pos[no] = (x, -prof * PASSO_Y)
            return x

        rec(0, 0)

        return pos, tipo, arestas

    def _info_no(self, t, no):

        v = t.value[no][0].astype(float)
        total = v.sum()
        p = v / total if total > 0 else v

        idx = int(np.argmax(p))
        classe = self.classes[idx] if idx < len(self.classes) else "?"

        # Versoes novas do scikit-learn guardam proporcoes; plot_tree mostra pesos.
        valores = v * t.weighted_n_node_samples[no] if abs(total - 1) < 1e-6 else v

        gini = 1 - float(np.sum(p ** 2))

        return {
            "p": p,
            "idx": idx,
            "classe": classe,
            "n": int(t.n_node_samples[no]),
            "gini": gini,
            "valores": valores,
        }

    def _texto_e_cor(self, t, no, tipo):

        info = self._info_no(t, no)
        eh_folha = tipo == "folha"
        cor_classe = COR_ACAO.get(info["classe"], "#888888")

        if self.modo == "didatico":

            if eh_folha:

                acao = TEXTO_ACAO.get(info["classe"], info["classe"])

                texto = (
                    f"{textwrap.fill(acao, 15)}\n"
                    f"{info['n']} jogadas\n"
                    f"confiança {info['p'][info['idx']] * 100:.0f}%"
                )

                return texto, clarear(cor_classe, 0.25)

            feat = self.colunas[t.feature[no]]
            texto_pergunta, _ = pergunta(feat, float(t.threshold[no]))

            texto = f"{textwrap.fill(texto_pergunta, 15)}\n{info['n']} jogadas"

            return texto, (0.97, 0.97, 0.98)

        # ----- modo tecnico (parecido com plot_tree) -----

        linhas = []

        if not eh_folha:
            feat = self.colunas[t.feature[no]]
            linhas.append(f"{feat} <= {float(t.threshold[no]):g}")

        valores = ", ".join(f"{x:.0f}" for x in info["valores"])

        linhas += [
            f"gini = {info['gini']:.3f}",
            f"samples = {info['n']}",
            f"value = [{valores}]",
            f"class = {info['classe']}",
        ]

        pureza = 1 - info["gini"] / max(1 - 1 / max(len(self.classes), 2), 1e-9)
        pureza = float(np.clip(pureza, 0, 1))

        return "\n".join(linhas), clarear(cor_classe, 0.15 + 0.7 * (1 - pureza))

    def _desenhar_arvore(self, recentralizar):

        limites = (self.ax.get_xlim(), self.ax.get_ylim())

        ax = self.ax
        ax.clear()
        ax.axis("off")
        ax.set_aspect("equal", adjustable="datalim")

        self.textos = []
        self.patches = {}
        self.estilo_padrao = {}
        self.arestas = {}

        t = self._arvore().tree_
        pos, tipo, arestas = self._layout(t)

        # ----- Arestas -----

        for pai, filho, eh_esq in arestas:

            xp, yp = pos[pai]
            xc, yc = pos[filho]

            linha, = ax.plot(
                [xp, xc], [yp - ALT / 2, yc + ALT / 2],
                color="#888888", linewidth=1.0, zorder=1,
            )

            self.arestas[(pai, filho)] = linha

            if self.modo == "didatico":

                feat = self.colunas[t.feature[pai]]
                _, sim_esq = pergunta(feat, float(t.threshold[pai]))
                sim = eh_esq == sim_esq
                rotulo = "Sim" if sim else "Não"
                cor = "#2a8a2a" if sim else "#b22222"

            else:

                rotulo = "True" if eh_esq else "False"
                cor = "#555555"

            txt = ax.text(
                (xp + xc) / 2, (yp + yc) / 2, rotulo,
                ha="center", va="center", color=cor, weight="bold",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9),
                zorder=3,
            )

            self.textos.append((txt, "aresta"))

        # ----- Caixas -----

        for no, (x, y) in pos.items():

            texto, cor = self._texto_e_cor(t, no, tipo[no])

            estilo = (
                "#666666",
                1.0,
                (0, (4, 3)) if tipo[no] == "cortado" else "-",
            )

            caixa = FancyBboxPatch(
                (x - LARG / 2, y - ALT / 2), LARG, ALT,
                boxstyle="round,pad=0,rounding_size=0.08",
                facecolor=cor, edgecolor=estilo[0], linewidth=estilo[1],
                linestyle=estilo[2], zorder=2,
            )

            ax.add_patch(caixa)

            self.patches[no] = caixa
            self.estilo_padrao[no] = estilo

            txt = ax.text(
                x, y, texto, ha="center", va="center",
                color="#111111", zorder=4, linespacing=1.15,
            )

            self.textos.append((txt, "no"))

            if tipo[no] == "cortado":

                marcador = ax.text(
                    x, y - ALT / 2 - 0.17, "▼ mais níveis",
                    ha="center", va="center", color="#777777", zorder=4,
                )

                self.textos.append((marcador, "aresta"))

        # ----- Enquadramento -----

        if recentralizar:

            xs = [p[0] for p in pos.values()]
            ys = [p[1] for p in pos.values()]

            ax.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
            ax.set_ylim(min(ys) - 1.0, max(ys) + 1.0)

        else:

            ax.set_xlim(*limites[0])
            ax.set_ylim(*limites[1])

        n_cortados = sum(1 for k in tipo.values() if k == "cortado")

        prof_max = t.tree_.max_depth if hasattr(t, "tree_") else None
        prof_max = self._arvore().tree_.max_depth

        self.txt_info.set_text(
            f"Árvore {self.indice + 1} de {self.n_arvores}   ·   "
            f"níveis mostrados: {min(self.nivel, max(prof_max, 1))} de {prof_max}   ·   "
            f"{len(pos)} caixas na tela"
            + (f"   ·   {n_cortados} com mais níveis embaixo" if n_cortados else "")
        )

        self._ajustar_fontes()
        self.fig.canvas.draw_idle()

    def _ajustar_fontes(self):
        """O texto acompanha o tamanho das caixas: zoom = letras maiores."""

        self.ax.apply_aspect()

        x0, x1 = self.ax.get_xlim()
        pixels_por_unidade = self.ax.bbox.width / max(x1 - x0, 1e-9)

        caracteres = 15 if self.modo == "didatico" else 24

        fonte = LARG * pixels_por_unidade / (caracteres * 0.62) * 72 / self.fig.dpi
        fonte = min(fonte, 13)

        visivel = fonte >= 3.5

        for txt, tipo in self.textos:

            txt.set_visible(visivel)

            if visivel:
                txt.set_fontsize(fonte if tipo == "no" else fonte * 0.9)

    # ==================================================
    # CENARIO, CAMINHO E VOTOS
    # ==================================================

    def _atualizar_rotulos_cenario(self):

        f = int(self.s1.val)
        e = int(self.s2.val)

        if f == 0:
            self.txt_s1.set_text("Fruta: alinhada (nem à frente nem atrás)")
        elif f > 0:
            self.txt_s1.set_text(f"Fruta: {f} casa(s) à FRENTE")
        else:
            self.txt_s1.set_text(f"Fruta: {-f} casa(s) ATRÁS")

        if e == 0:
            self.txt_s2.set_text("Fruta: alinhada (nem à esquerda nem à direita)")
        elif e > 0:
            self.txt_s2.set_text(f"Fruta: {e} casa(s) à ESQUERDA")
        else:
            self.txt_s2.set_text(f"Fruta: {-e} casa(s) à DIREITA")

        self.txt_s3.set_text(f"Tamanho da cobra: {int(self.s3.val)} casa(s)")

    def _vetor_cenario(self):

        perigo = self.chk.get_status()

        valores = {
            "fruta_frente": int(self.s1.val),
            "fruta_esq": int(self.s2.val),
            "rel_perigo_frente": int(perigo[0]),
            "rel_perigo_esq": int(perigo[1]),
            "rel_perigo_dir": int(perigo[2]),
            "TamanhoCobrinha": int(self.s3.val),
        }

        return np.array([[valores[c] for c in self.colunas]], dtype=np.float32)

    def _ao_mudar_cenario(self, _valor):

        self._atualizar_rotulos_cenario()
        self._atualizar_cenario()

    def _destacar(self, x):

        arvore = self._arvore()
        caminho = list(arvore.decision_path(x).indices)
        pares = set(zip(caminho[:-1], caminho[1:]))

        no_caminho = set(caminho)

        for no, caixa in self.patches.items():

            if no in no_caminho:
                caixa.set_edgecolor("#d62728")
                caixa.set_linewidth(3.2)
                caixa.set_linestyle("-")
                caixa.set_zorder(3)
            else:
                cor, largura, estilo = self.estilo_padrao[no]
                caixa.set_edgecolor(cor)
                caixa.set_linewidth(largura)
                caixa.set_linestyle(estilo)
                caixa.set_zorder(2)

        for chave, linha in self.arestas.items():

            if chave in pares:
                linha.set_color("#d62728")
                linha.set_linewidth(3.0)
            else:
                linha.set_color("#888888")
                linha.set_linewidth(1.0)

    def _atualizar_cenario(self):

        x = self._vetor_cenario()

        self._destacar(x)

        # ----- Votos de cada arvore -----

        votos = np.zeros(len(self.classes), dtype=int)

        for arvore in self.modelo.estimators_:
            votos[int(arvore.predict(x)[0])] += 1

        voto_da_tela = int(self._arvore().predict(x)[0])

        decisao = self.modelo.predict(pd.DataFrame(x, columns=self.colunas))[0]

        ax = self.ax_votos
        ax.clear()

        ax.bar(
            range(len(self.classes)),
            votos,
            color=[COR_ACAO.get(c, "#888888") for c in self.classes],
        )

        for i, v in enumerate(votos):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=10)

        ax.set_xticks(range(len(self.classes)))
        ax.set_xticklabels(self.classes, fontsize=9.5)
        ax.set_ylim(0, max(votos.max() * 1.18, 1))
        ax.set_title(f"Votos das {self.n_arvores} árvores", fontsize=10.5)

        ax.set_xlabel(
            f"A árvore {self.indice + 1} (na tela) vota:\n"
            f"{TEXTO_ACAO.get(self.classes[voto_da_tela], self.classes[voto_da_tela])}",
            fontsize=9,
        )

        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)

        self.txt_dec.set_text(
            f"A floresta decide:\n{TEXTO_ACAO.get(str(decisao), str(decisao))}"
        )
        self.txt_dec.set_color(COR_ACAO.get(str(decisao), "#333333"))

        self.fig.canvas.draw_idle()

    # ==================================================
    # NAVEGACAO
    # ==================================================

    def _trocar_arvore(self, passo):

        self.indice = (self.indice + passo) % self.n_arvores

        self._desenhar_arvore(recentralizar=True)
        self._atualizar_cenario()

    def _mudar_nivel(self, delta):

        prof_max = max(1, self._arvore().tree_.max_depth)

        self.nivel = int(np.clip(self.nivel + delta, 1, prof_max))

        self._desenhar_arvore(recentralizar=True)
        self._atualizar_cenario()

    def _centralizar(self):

        self._desenhar_arvore(recentralizar=True)
        self._atualizar_cenario()

    def _alternar_modo(self):

        self.modo = "tecnico" if self.modo == "didatico" else "didatico"

        self._atualizar_botao_modo()
        self._atualizar_ajuda()

        self._desenhar_arvore(recentralizar=False)
        self._atualizar_cenario()

    def _atualizar_botao_modo(self):

        proximo = "Técnico" if self.modo == "didatico" else "Didático"
        atual = "Didático" if self.modo == "didatico" else "Técnico"

        self.b_modo.label.set_text(f"Modo: {atual}  (clique para ver o {proximo})")
        self.b_modo.label.set_fontsize(10)

    def _atualizar_ajuda(self):

        if self.modo == "didatico":

            texto = (
                "Cada caixa clara é uma PERGUNTA sobre o jogo. Siga o Sim ou o Não até chegar "
                "numa caixa COLORIDA: essa é a decisão desta árvore. A floresta junta as "
                f"{self.n_arvores} árvores e vence a ação mais votada."
            )

        else:

            texto = (
                "Modo técnico: gini mede a mistura de ações na caixa (0 = só uma ação); "
                "samples = nº de jogadas que chegaram ali; "
                f"value = peso de cada ação [{', '.join(self.classes)}]; "
                "class = ação mais provável. True = ramo da esquerda, False = ramo da direita."
            )

        self.txt_ajuda.set_text(textwrap.fill(texto, 150))

    # ==================================================
    # MOUSE E TECLADO
    # ==================================================

    def _barra_ativa(self):

        barra = getattr(self.fig.canvas, "toolbar", None)

        return bool(getattr(barra, "mode", ""))

    def _ao_rolar(self, ev):

        if ev.inaxes is not self.ax or ev.xdata is None:
            return

        fator = 0.8 if ev.button == "up" else 1.25

        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        xd, yd = ev.xdata, ev.ydata

        self.ax.set_xlim(xd - (xd - x0) * fator, xd + (x1 - xd) * fator)
        self.ax.set_ylim(yd - (yd - y0) * fator, yd + (y1 - yd) * fator)

        self._ajustar_fontes()
        self.fig.canvas.draw_idle()

    def _ao_apertar(self, ev):

        if ev.inaxes is self.ax and ev.button == 1 and not self._barra_ativa():
            self.arrasto = (ev.xdata, ev.ydata)

    def _ao_soltar(self, _ev):

        self.arrasto = None

    def _ao_mover(self, ev):

        if self.arrasto is None or ev.inaxes is not self.ax or ev.xdata is None:
            return

        dx = self.arrasto[0] - ev.xdata
        dy = self.arrasto[1] - ev.ydata

        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()

        self.ax.set_xlim(x0 + dx, x1 + dx)
        self.ax.set_ylim(y0 + dy, y1 + dy)

        self.fig.canvas.draw_idle()

    def _ao_tecla(self, ev):

        if ev.key == "right":
            self._trocar_arvore(+1)
        elif ev.key == "left":
            self._trocar_arvore(-1)
        elif ev.key == "up":
            self._mudar_nivel(+1)
        elif ev.key == "down":
            self._mudar_nivel(-1)
        elif ev.key in ("m", "M"):
            self._alternar_modo()

    def _ao_redimensionar(self, _ev):

        self._ajustar_fontes()

    # ==================================================
    # SEM JANELA: SALVAR IMAGENS
    # ==================================================

    def salvar_imagens(self):

        caminhos = []

        for modo, nome in (
            ("didatico", "arvore_didatica.png"),
            ("tecnico", "arvore_tecnica.png"),
        ):

            self.modo = modo

            self._atualizar_botao_modo()
            self._atualizar_ajuda()
            self._desenhar_arvore(recentralizar=True)
            self._atualizar_cenario()

            caminho = os.path.join(PASTA_PROJETO, nome)
            self.fig.savefig(caminho, dpi=110)

            caminhos.append(caminho)

        return caminhos


def mostrar_arvore(modelo, colunas, abrir_janelas=True, **_ignorados):

    visualizador = Visualizador(modelo, colunas)

    if abrir_janelas and plt.get_backend().lower() != "agg":

        print("\nJanela interativa aberta: feche-a para encerrar o programa.")
        print("  - Botões e setas: trocar de árvore (← →) e mostrar mais níveis (↑ ↓)")
        print("  - Arraste para mover e use a roda do mouse para dar zoom")
        print("  - Botão 'Modo' (ou tecla M): alterna entre didático e técnico")

        plt.show()

    else:

        caminhos = visualizador.salvar_imagens()

        print("\nSem janela gráfica: imagens salvas em")

        for caminho in caminhos:
            print(" ", caminho)

        plt.close("all")
