# Changelog

Todas as alterações notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado no [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/) e este projeto adere ao [Versionamento Semântico](https://semver.org/lang/pt-BR/).

## [0.3.11] - 2026-10-13

### Adicionado
- **Filtro One-Euro 2D Isotrópico (`OneEuroFilter2D`):** Cálculo de corte adaptativo baseado na velocidade vetorial conjunta 2D ($v = \|\mathbf{v}\|_2$), garantindo invariância rotacional estrita e eliminando distorções de trajetória e atraso de eixo em sacadas oblíquas, com suporte a prealocação in-place (`out=...`).
- **Estabilizador com Âncora Contínua e Convergência ao Alvo (`FixationStabilizer`):** Modelo híbrido $C^0$ que elimina o efeito catraca de 45 px E implementa convergência assintótica suave (*catch-up*) ao alvo no repouso ($v_{\text{gaze}} < 40\text{ px/s}$), eliminando a histerese permanente de arrasto (offset $< 0,005\text{ px}$) e mantendo jitter de fixação em $1,95\text{ px RMS}$.
- **Proteção Anti-Equívoco de Comandos de Gestos (`CursorSession`):** Debounce temporal consciente ($\ge 2$ quadros ou $\ge 50\text{ ms}$) com isolamento do 1º quadro candidato de fechamento palpebral (previne solavancos causados pela íris distorcida pela pálpebra) e resiliência a quedas severas de FPS ($\le 10$ FPS).
- **Snapshot Imutável em `PositionHistory`:** Gravação desacoplada de coordenadas em tuplas imutáveis e retorno de cópias seguras, garantindo integridade de rollback pré-gesto contra mutações in-place.
- **Suíte de Testes Automatizados:** 13 novos testes unitários e de integração em `test_filters.py` e `test_tracking_smoothing_30fps.py` cobrindo isotropia, atenuação de jitter a 30 FPS, continuidade de âncora, convergência a offset zero pós-perseguição, isolamento de quadro candidato, resiliência a 10 FPS e imutabilidade de histórico (82/82 testes aprovados, 100%).
- **Relatório Técnico Detalhado:** Publicação de [`reports/RELATORIO_TECNICO_SUAVIZACAO_TRACKING_30FPS.md`](file:///C:/Users/Herna/OneDrive/Documentos/workspace/L.O.R.E.N.A.-DESKTOP/reports/RELATORIO_TECNICO_SUAVIZACAO_TRACKING_30FPS.md) com deduções matemáticas, provas de continuidade e métricas empíricas.

### Modificado
- **Zero Alocações no Loop Crítico a 30 FPS:** Reutilização estrita de buffers pré-alocados `_buf_euro` e `_buf_dispatch`, e substituição de `np.linalg.norm` por `math.hypot`.
- **Configurações Tipadas (`lorena/config.py` e `settings/config.json`):** Adição de `isotropic_smoothing`, `smooth_deadzone`, `break_deadzone_mult`, `freeze_confirm_frames` e `catchup_rate` na dataclass `FilterSettings`.

---

## [0.3.10] - 2026-10-13

### Corrigido
- **NameError no Ponto de Entrada (`lorena/app.py`):** Adicionada a definição explícita do logger `log = logging.getLogger(__name__)` em `lorena/app.py`, eliminando o erro de inicialização `NameError: name 'log' is not defined` ao disparar a aplicação via `eyemouse.py`, `eye2mouse.py` ou `python -m lorena`.
- **Isolamento de Recursos Tkinter (`MainWindow`):** Vinculação explícita de `PhotoImage(master=r)` e encapsulamento com tratamento defensivo em `r.iconphoto` para prevenir falhas de contexto entre múltiplas janelas ou instâncias em suítes de teste.

### Adicionado
- **Padrões de Engenharia de Testes com Pytest (`AGENTS.md`):** Nova seção formal contendo pirâmide de testes DDD, diretrizes para testes de modelos biomecânicos e concorrentes, anti-padrões proibidos (incluindo eliminação de bypasses de pytest em código de produção) e checklist de DoD de testes.
- **Fixtures Canônicas Globais (`tests/conftest.py`):** Criação de fixtures padronizadas (`synthetic_frame`, `dummy_face_landmarks`, `sample_gaze_features`, `sample_calibration_dataset`, `mock_cursor`, `tk_root` com teardown seguro via `yield`).
- **Marcadores Semânticos de Teste (`pyproject.toml`):** Configuração de marcadores para categorização de testes (`unit`, `integration`, `ui`, `hardware`, `slow`).
- **Suíte de Testes para o Composition Root (`tests/test_app.py`):** 2 novos testes automatizados cobrindo a existência do logger do módulo e o ciclo de instanciação limpa de `app.main()` (69/69 testes passando, 100%).

---

## [0.3.9] - 2026-10-13

### Adicionado
- **Auto-Promoção da Câmera RGB (`MainWindow`):** Seleção inteligente que detecta se a fonte padrão é um sensor IR (comum em notebooks com Windows Hello) e prioriza automaticamente a webcam RGB disponível.
- **Desativação Segura de Botões em Lista Vazia (`MainWindow`):** Desativação automática de botões de Play e Calibração quando nenhuma fonte de captura está disponível, prevenindo falhas de inicialização.
- **Botão de Atualização Sob Demanda (`MainWindow`):** Botão "🔄" ao lado do combobox de fontes para redetecção instantânea de câmeras USB conectadas em tempo de execução (hotplug).
- **Tolerância a Perda de Sinal e Alerta Visual (`TrackingPipeline` e `StatusOverlay`):** Tolerância de até 3 segundos com tentativas contínuas de leitura em caso de interrupção do sinal de vídeo, acompanhado de alerta visual imediato ("Câmera sem sinal — reconectando...") no overlay superior.
- **Identificação do Backend em Log (`Camera.open`):** Registro explícito do backend do OpenCV via `cap.getBackendName()` para auditoria de performance.
- **Suíte de Testes Automatizados:** 4 novos testes adicionados em `test_camera.py` e `test_ui_main_window.py` cobrindo classificação de IR sob alta luz, interrupção contígua de probing, auto-promoção RGB e atualização a quente (67/67 testes passando, 100%).

### Corrigido
- **Eliminação de Bloqueio da UI Thread (`MainWindow`):** Remoção de todas as chamadas síncronas de `time.sleep()` nos métodos de evento Tkinter (`_on_play`, `_stop_tracking`, `_on_calibrate`, `_on_calibration_finished`), transferindo todo o tratamento de sincronização para as threads secundárias (0 ms de overhead de UI, cumprindo o SLA do sistema).
- **Aceleração do Probing de Câmeras (`list_available_cameras`):** Interrupção precoce (`break`) na sondagem ao falhar o primeiro índice contíguo, reduzindo o tempo de varredura em mais de 50%.
- **Eliminação de Subprocesso Pesado:** Remoção da chamada externa síncrona do PowerShell que causava atraso de 1,8 s na inicialização.
- **Priorização Estrita de Backend MSMF (`Camera.open` e `CameraPreview`):** Isolamento de retries de `cv2.CAP_MSMF` antes de permitir qualquer fallback para `CAP_DSHOW`, prevenindo rebaixamento indevido de framerate para 10-14 FPS por limitação de largura de banda YUY2.
- **Classificação Cromática Rigorosa:** Correção da lógica booleana em `list_available_cameras`, exigindo saturação e variação inter-canal para rotular fontes como RGB, e rótulo claro de `"Sensor IR ou Tampa Fechada"` quando a luminosidade for nula.
- **Eliminação de Leitura Dupla de Quadros:** Sondagem de dispositivos otimizada para leitura única de quadro por índice.

### Modificado
- **Otimização de Processamento de Memória:** Subamostragem 8x8 (`np.int16`) no cálculo de croma (redução de 64x em alocação de memória) e inversão da ordem de operações no `CameraPreview._worker()` aplicando `cv2.resize` antes de `cv2.flip` (redução de 92% nas operações de pixels por quadro).

---

## [0.3.8] - 2026-10-13

### Adicionado
- **Classificação e Preservação de Dispositivos (`lorena.vision.camera`):** Diferenciação dinâmica de câmeras entre `RGB / Colorida` e `Sensor IR / Monocromático`, priorizando sensores RGB no topo da lista sem descartar sensores secundários ou de baixa luminosidade.
- **Resiliência com Backoff em Hardware (`Camera.open`):** Laço de até 3 tentativas com pausa de 150 ms para absorver a latência de liberação assíncrona de drivers no Windows Media Foundation (`CAP_MSMF`).
- **Tratamento de Exceções de Matriz (`cv2.error`):** Tratamento defensivo em `Camera._capture_worker`, `CameraPreview._worker` e `TrackingPipeline._run` para recuperar automaticamente de asserções `_step >= minstep` causadas por corrupção transitória de buffer.
- **Suíte de Testes Automatizados:** 5 novos testes unitários e de integração em `tests/test_camera.py` e `tests/test_ui_main_window.py` cobrindo classificação de fontes, repetição de abertura e ciclo de vida de transição UI $\leftrightarrow$ pipeline.

### Corrigido
- **Sobrescrita Acidental de Configuração:** Correção na inicialização de `MainWindow` para evitar descarte da Câmera 0 e sobrescrita indevida do índice em `settings/config.json`.
- **Conflito de Dispositivo (*Device Busy*):** Eliminação do popup modal `messagebox.showinfo` no caminho crítico de `_on_play()`, garantindo que o `CameraPreview` seja encerrado e seu handle liberado 150 ms antes de inicializar o `TrackingPipeline`.
- **Restauração Limpa de Preview:** Retomada automática e temporizada do preview ao término do rastreio ocular (ESC) ou conclusão de calibração.

### Modificado
- **Interface Gráfica (`MainWindow`):** Combobox ajustado para largura expandida (`width=28`) com rótulos informativos de capacidade de sensor e feedback de logging estruturado no arranque da aplicação.
