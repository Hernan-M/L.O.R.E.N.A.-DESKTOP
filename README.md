# L.O.R.E.N.A. — Eye Tracking Mouse

Assistente virtual para controle de cursor de mouse por rastreio ocular em Python, orientado a objetos, de alta precisão e baixa latência.

---

## 🎯 Principais Características

- **Detecção Direta de Íris Subpixel:** Utiliza MediaPipe Tasks `FaceLandmarker` em modo vídeo contínuo (sem conversão para dlib ou limiarização binária).
- **Normalização Ocular Invariante:** Projeção vetorial nos eixos anatômicos do olho, estável contra rotações de cabeça (roll), variação de distância e movimentos da pálpebra.
- **Mapeamento por Regressão Polinomial com Regularização Ridge:** Modela curvatura do globo ocular e acoplamento horizontal/vertical com validação Leave-One-Out.
- **Filtragem Tripla de Alta Estabilidade:**
  - *Outlier Gate:* rejeita picos espúrios isolados de 1 frame.
  - *One Euro Filter 2D:* filtro adaptativo que elimina tremor em fixações e mantém resposta instantânea em sacadas.
  - *Fixation Stabilizer:* zona morta dinâmica para ancoragem estável em alvos.
- **Gestos de Clique e Acessibilidade:**
  - Máquina de estados com histerese baseada em EAR (Eye Aspect Ratio) normalizado.
  - Ignora piscadas naturais bilaterais espontâneas.
  - Wink de olho esquerdo para clique padrão e wink mantido para clique direito.
  - Congelamento com recuo temporal (*rollback*) de 150 ms para garantir clique exatamente no alvo.
  - Suporte a *Dwell Click* (clique por tempo de permanência de fixação).
  - Pausa/retomada de rastreio mantendo ambos os olhos fechados por 1.2s.
- **Integração Win32 de Baixa Latência e High-DPI:**
  - Envio de cursor sem `sleep` via Win32 `SetCursorPos`.
  - ESC global de emergência detectado em segundo plano mesmo sem janela ativa.
  - Tratamento nativo de High-DPI para monitores com escala (125%, 150%).

---

## 📋 Pré-requisitos e Instalação

Requer **Python 3.10+**.

```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente virtual (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Instalar dependências
pip install -r requirements.txt
```

---

## 🚀 Como Executar

### Interface Gráfica Principal
```bash
python -m lorena
# ou
python eye2mouse.py
```

1. **Calibrar:** Clique em **Calibrar**. Olhe fixamente para cada um dos 9 pontos na tela até que o círculo encolha e fique verde. A calibração é 100% automática (não requer cliques manuais).
2. **Iniciar:** Clique em **Play**.
3. **Sair:** Pressione a tecla **ESC** a qualquer momento para finalizar o rastreio.

### Ferramenta de Depuração Visual
Para visualizar os marcos anatômicos, vetores de olhar e métricas em tempo real:
```bash
python tools/debug_view.py
```

### Executar Testes Automatizados
```bash
pytest -v
```

---

## 📖 Documentação Técnica e Roadmap

Consulte [`docs/PLANO_DE_MELHORIAS.md`](docs/PLANO_DE_MELHORIAS.md) para o diagnóstico matemático detalhado dos erros do código legado, fórmulas implementadas, arquitetura e roadmap completo de evolução.
